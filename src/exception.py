"""
Custom exception handling for Diamond Price Predictor.
"""

import sys
import logging
from typing import Optional


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generate detailed error message with file and line information.
    
    Args:
        error: The exception that occurred
        error_detail: System error details
        
    Returns:
        Formatted error message string
    """
    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script [{file_name}] "
            f"at line number [{line_number}] with error message [{str(error)}]"
        )
    else:
        error_message = f"Error occurred: {str(error)}"
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class for Diamond Price Predictor with detailed error reporting.
    """
    
    def __init__(self, error_message: str, error_detail: Optional[sys] = None):
        """
        Initialize custom exception.
        
        Args:
            error_message: Error message or exception
            error_detail: System error details
        """
        super().__init__(error_message)
        
        if error_detail is None:
            error_detail = sys
            
        if isinstance(error_message, Exception):
            self.error_message = error_message_detail(error_message, error_detail)
        else:
            self.error_message = error_message
        
        # Log the error
        logging.error(self.error_message)
    
    def __str__(self):
        """Return string representation of the exception."""
        return self.error_message