import sys


def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    return (
        f"Error occurred in Python script\n"
        f"File: {exc_tb.tb_frame.f_code.co_filename}\n"
        f"Line: {exc_tb.tb_lineno}\n"
        f"Message: {str(error)}"
    )


class CustomException(Exception):

    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

    def __str__(self):
        return self.error_message