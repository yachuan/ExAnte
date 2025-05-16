import os
import logging
import csv
import re
from datetime import datetime

class FileUtils:
    @staticmethod
    def setup_logging(output_dir, title):
        log_dir = os.path.join(output_dir, title, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger = logging.getLogger(title)
        logger.setLevel(logging.INFO)
        logger.handlers = []  # Clear any existing handlers
        logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def get_content_path(content_dir, title, year):
        """Get the correct content file path for a given title directory and year"""
        return os.path.join(content_dir, f'{title}_{year}_content.txt')

    @staticmethod
    def read_content(file_path, logger):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully read content from {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def create_csv_file(folder_path, title, strategy):
        csv_path = os.path.join(folder_path, f'{title}_{strategy}.csv')
        headers = [
            'Atomic Fact',
            'Judgment before Cutoff vs Wiki Before',
            'Judgment before Cutoff vs Wiki After',
            'Initial Leakage Check',
            'Reference Before',
            'Reference After',
            'Response from Judger Model Before',
            'Response from Judger Model After',
            'Derivation Analysis',
            'Final Leakage'
        ]
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return csv_path
        except Exception as e:
            return None
    
    @staticmethod
    def get_cutoff_year(content_dir):
        """Get the cutoff year from a content directory"""
        try:
            # Get list of content files in the directory
            content_files = [f for f in os.listdir(content_dir) if f.endswith('_content.txt')]
            
            # Extract years from filenames
            years = []
            for filename in content_files:
                match = re.search(r'_(\d{4})_content\.txt$', filename)
                if match:
                    years.append(int(match.group(1)))
            
            if not years:
                print(f"No valid years found in {content_dir}, defaulting to 2024")
                return '2024'
            
            cutoff_year = str(min(years))
            print(f"Found cutoff year {cutoff_year} from files: {content_files}")
            return cutoff_year
            
        except Exception as e:
            print(f"Error getting cutoff year from {content_dir}: {str(e)}")
            return '2024' 