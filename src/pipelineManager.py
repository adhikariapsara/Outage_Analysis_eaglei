
import os
import logging

from networkx import config

from src import eaglei_cleaning
from src import weather_cleaning
from src import fetch_weather_data
from src import merge_outage_weather
from src import multi_county_analysis
from src import calculate_event_stats
from src.configManager import ConfigManager

logger = logging.getLogger("eagleiLogger")

# ==================== Pipeline Components ====================
class PipelineManager:
    """Manages the execution of the EAGLE-i analysis pipeline."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize PipelineManager.
        
        Args:
            config: ConfigManager instance with loaded configuration
        """
        self.config = config
        self.params = config.get_processing_parameters()
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.config.get("data_paths.outage_data_dir"),
            self.config.get("data_paths.weather_data_dir"),
            self.config.get("data_paths.results_dir"),
            self.config.get("data_paths.event_data_dir"),
            "logs"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_full_pipeline(self) -> bool:
        """
        Execute the complete analysis pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 70)
            logger.info("Starting EAGLE-i Data Processing Pipeline")
            logger.info("=" * 70)
            
            state = self.config.get("processing_parameters.state")
            county = self.config.get("processing_parameters.county")
            start = self.config.get("processing_parameters.start_year")
            end = self.config.get("processing_parameters.end_year")
            
            # Step 1: Fetch weather data
            if self.config.get("processing_options.fetch_weather_data"):
                logger.info(f"Step 1/5: Fetching weather data for {state}...")
                self._fetch_weather(state, county, start, end)
            
            # Step 2: Clean weather data
            if self.config.get("processing_options.clean_weather_data"):
                logger.info(f"Step 2/5: Cleaning weather data...")
                self._clean_weather(state, county, start, end)
            
            # Step 3: Clean outage data
            if self.config.get("processing_options.clean_outage_data"):
                logger.info(f"Step 3/5: Cleaning outage data...")
                self._clean_outage(state, county, start, end)
            
            # Step 4: Align and merge data
            if self.config.get("processing_options.align_data"):
                logger.info(f"Step 4/5: Aligning outage and weather data...")
                self._align_data(state, county, start, end)

            # Step 5: Calculate weather/outage stats for each event
            if self.config.get("processing_options.calculate_event_stats"):
                logger.info(f"Step 5/5: Calculating event stats...")
                self._calculate_event_stats_single_county(state, county, start, end)
            
            logger.info("=" * 70)
            logger.info("✓ Full Pipeline Completed Successfully!")
            logger.info("=" * 70)
            return True
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return False
    
    def run_outage_only(self) -> bool:
        """Execute only outage data processing."""
        try:
            logger.info("Starting EAGLE-i Outage Data Processing...")
            state = self.config.get("processing_parameters.state")
            county = self.config.get("processing_parameters.county")
            start = self.config.get("processing_parameters.start_year")
            end = self.config.get("processing_parameters.end_year")
            
            self._clean_outage(state, county, start, end)
            logger.info("✓ Outage processing completed!")
            return True
        except Exception as e:
            logger.error(f"Outage processing failed: {e}", exc_info=True)
            return False
    
    def run_weather_only(self) -> bool:
        """Execute only weather data processing."""
        try:
            logger.info("Starting Weather Data Processing...")
            state = self.config.get("processing_parameters.state")
            county = self.config.get("processing_parameters.county")
            start = self.config.get("processing_parameters.start_year")
            end = self.config.get("processing_parameters.end_year")
            
            self._fetch_weather(state, county, start, end)
            self._clean_weather(state, county, start, end)
            logger.info("✓ Weather processing completed!")
            return True
        except Exception as e:
            logger.error(f"Weather processing failed: {e}", exc_info=True)
            return False
    
    def run_alignment_only(self) -> bool:
        """Execute only data alignment."""
        try:
            logger.info("Starting Data Alignment...")
            state = self.config.get("processing_parameters.state")
            county = self.config.get("processing_parameters.county")
            start = self.config.get("processing_parameters.start_year")
            end = self.config.get("processing_parameters.end_year")
            
            self._align_data(state, county, start, end)
            logger.info("✓ Data alignment completed!")
            return True
        except Exception as e:
            logger.error(f"Data alignment failed: {e}", exc_info=True)
            return False

        
    def run_multi_county_analysis(self) -> bool:
        """Execute multi-county analysis."""
        try:
            logger.info("Starting Multi-County Analysis...")
            county_pairs = self.config.get("multi_county_analysis_parameters.state_county_pairs")
            start = self.config.get("processing_parameters.start_year")
            end = self.config.get("processing_parameters.end_year")
            
            # first check if county_pairs is not empty
            if not county_pairs:
                logger.error("No county pairs specified for multi-county analysis.")
                return False
            
            # check if the required files exist for each county pair in the merged data directory
            for pair in county_pairs:
                state = pair["state"]
                county = pair["county"]
                merged_file_dir = os.path.join(self.config.get("data_paths.merged_data_dir"), state)
                merged_file_name = self.config.get("file_patterns.merged_file_pattern").format(start=start, end=end, county=county, state=state)
                merged_file_path = os.path.join(merged_file_dir, merged_file_name)
                
                if not os.path.isfile(merged_file_path):
                    logger.error(f"Merged data file {merged_file_path} not found. Please run the full pipeline for {state} - {county} first.")
                    return False

            multi_county_analysis.create_multi_county_events(county_pairs, start, end, self.config)

            if self.config.get("processing_options.calculate_event_stats"):
                self._calculate_event_stats_multi_county(start, end)

            logger.info("✓ Multi-County Analysis Completed Successfully!")
            return True
        except Exception as e:
            logger.error(f"Multi-County Analysis failed: {e}", exc_info=True)
            return False   

    def _fetch_weather(self, state: str, county: str, start: int, end: int) -> None:
        """Fetch weather data with caching."""
        weather_file_dir = os.path.join(self.config.get("data_paths.weather_data_dir"), state)
        weather_file_name = self.config.get("file_patterns.weather_file_pattern").format(state=state, county=county, start=start, end=end)
        weather_file = os.path.join(weather_file_dir, weather_file_name)
        
        if os.path.isfile(weather_file) and self.config.get("processing_options.skip_if_exists"):
            logger.info(f"  → Weather data already exists. Skipping.")
            return
        
        logger.info(f"  → Fetching weather data from IEM servers...")
        # check if the directory exists, if not create it
        os.makedirs(weather_file_dir, exist_ok=True)
        fetch_weather_data.main(state, county, start, end, weather_file)
        logger.info(f"  ✓ Weather data fetched successfully")
    
    def _clean_weather(self, state: str, county: str, start: int, end: int) -> None:
        """Clean weather data with caching."""
        weather_file_dir = os.path.join(self.config.get("data_paths.weather_data_dir"), state)
        weather_file_name = self.config.get("file_patterns.weather_file_pattern").format(state=state, county=county, start=start, end=end)
        raw_weather_file = os.path.join(weather_file_dir, weather_file_name)
        cleaned_file_name = self.config.get("file_patterns.cleaned_weather_file_pattern").format(state=state, county=county, start=start, end=end)
        cleaned_file = os.path.join(weather_file_dir, cleaned_file_name)
        
        if os.path.isfile(cleaned_file) and self.config.get("processing_options.skip_if_exists"):
            logger.info(f"  → Cleaned weather data already exists. Skipping.")
            return
        
        # first check if the raw weather data file exists
        if not os.path.isfile(raw_weather_file):
            logger.error(f"Raw weather data file {raw_weather_file} not found. Download weather data first.")
            raise FileNotFoundError(f"Raw weather data file {raw_weather_file} not found.")
        
        logger.info(f"  → Cleaning weather data...")
        weather_vars = self.config.get("weather_variables").values()
        weather_cleaning.create_netcdf_weather_stations(state, county, start, end, raw_weather_file, cleaned_file, weather_vars, min_data_threshold=0.1)
        logger.info(f"  ✓ Weather data cleaned successfully")
    
    def _clean_outage(self, state: str, county: str, start: int, end: int) -> None:
        """Clean outage data with caching."""
        outage_file_dir = os.path.join(self.config.get("data_paths.outage_data_dir"), state)
        cleaned_outage_file_name = self.config.get("file_patterns.cleaned_outage_file_pattern").format(start=start, end=end, county=county, state=state)
        cleaned_outage_file = os.path.join(outage_file_dir, cleaned_outage_file_name)
        
        if os.path.isfile(cleaned_outage_file) and self.config.get("processing_options.skip_if_exists"):
            logger.info(f"  → Cleaned outage data already exists. Skipping.")
            return
        
        # first check if the raw outage data files exist for all years from start to end (inclusive)
        for year in range(start, end + 1):
            outage_file_name = self.config.get("file_patterns.outage_file_pattern").format(year=year)
            outage_file = os.path.join(self.config.get("data_paths.eaglei_data_dir"), outage_file_name)
            if not os.path.isfile(outage_file):
                logger.error(f"EAGLE-I outage data file {outage_file} not found. Cannot clean outage data.")
                raise FileNotFoundError(f"EAGLE-I outage data file {outage_file} not found.")
        
        logger.info(f"  → Cleaning outage data...")
        # check if the directory exists, if not create it
        os.makedirs(outage_file_dir, exist_ok=True)
        eaglei_cleaning.main(state, county, start, end, cleaned_outage_file, self.config)
        logger.info(f"  ✓ Outage data cleaned successfully")
    
    def _align_data(self, state: str, county: str, start: int, end: int) -> None:
        """Align and merge outage and weather data."""
        merged_file_dir = os.path.join(self.config.get("data_paths.merged_data_dir"), state)
        merged_file_name = self.config.get("file_patterns.merged_file_pattern").format(start=start, end=end, county=county, state=state)
        merged_file_path = os.path.join(merged_file_dir, merged_file_name)
        
        if os.path.isfile(merged_file_path) and self.config.get("processing_options.skip_if_exists"):
            logger.info(f"  → Merged data already exists. Skipping.")
            return
        
        logger.info(f"  → Aligning and Merging outage and weather datasets...")
        merge_outage_weather.merge_outages_with_weather_netcdf(state, county, start, end, self.config)
        logger.info(f"  ✓ Data alignment completed")
        logger.info(f"  → Results saved to: {self.config.get('data_paths.merged_data_dir')}/{state}/")

    def _calculate_event_stats_single_county(self, state: str, county: str, start: int, end: int) -> None:
        event_file_dir= os.path.join(self.config.get("data_paths.event_data_dir"), "county_events", state)
        event_file_name= self.config.get("file_patterns.event_stats_file_pattern").format(start=start, end=end, county=county)
        event_file_path = os.path.join(event_file_dir, event_file_name)

        if os.path.isfile(event_file_path) and self.config.get("processing_options.skip_if_exists"):
            logger.info(f"  → Event data already exists. Skipping.")
            return

        logger.info(f"  → Calculating Event Statistics...")
        calculate_event_stats.calculate_event_stats_single_county(state, county, start, end, self.config)
        logger.info(f"  ✓ Event Calculations completed")
        logger.info(f"  → Results saved to: {self.config.get('data_paths.event_data_dir')}/county_events/{state}")

    def _calculate_event_stats_multi_county(self, start: int, end: int) -> None:
        event_file_dir= os.path.join(self.config.get("data_paths.event_data_dir"), "spatiotemporal_events")
        event_file_name= self.config.get("file_patterns.event_stats_file_pattern").format(start=start, end=end,
                                                county=self.config.get("multi_county_analysis_parameters.label"))
        event_file_path = os.path.join(event_file_dir, event_file_name)
        if os.path.isfile(event_file_path) and self.config.get("processing_options.skip_if_exists"):
            logger.info(f"  → Event data already exists. Skipping.")
            return
        logger.info(f"  → Calculating Event Statistics...")
        calculate_event_stats.initialize_multi_county_calc(start, end, self.config)
        logger.info(f"  ✓ Event Calculations completed")
        logger.info(f"  → Results saved to: {self.config.get('data_paths.event_data_dir')}/spatiotemporal_events")