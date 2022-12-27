class TerritorialSelectError(Exception):
    """Exception raised when objects are absent whithin a given territory.

    Attributes:
        object - DataFrame with specified object
        message - explanation of the error
    """
    def __init__(self, object_name):
        self.object = object_name
        self.message = f"There is no {object_name} whithin a given territory."
        super().__init__(self.message)


class SelectedValueError(Exception):
    """Exception raised for errors in the selected value.

    Attributes:
        value - input value which caused the error
        column - column of DataFrame in which select must be taken by value
        object - DataFrame with specified object
        message - explanation of the error
    """
    def __init__(self, object_name, value, column):
        self.object_name = object_name
        self.value = value
        self.column = column
        self.message = f"There is no {object_name} with values '{value}' in column '{column}'."
        super().__init__(self.message)


class ImplementationError(Exception):
    """Exception raised for errors in case the call of the method 
    with the given parameters has not yet been implemented.

    Attributes:
        value - input value which caused the error
        column - column of DataFrame in which select must be taken by value
        object - DataFrame with specified object
        message - explanation of the error
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NormativeError(Exception):
    """Exception raised for errors in case when a required parameter inside a method
    is not defined not by a user nor by specification data.

    Attributes:
        value - input value which caused the error
        column - column of DataFrame in which select must be taken by value
        object - DataFrame with specified object
        message - explanation of the error
    """
    def __init__(self, value, object):
        self.value = value
        self.object = object
        self.message = f"For the '{object}' the '{value}' parameter should be defined by user."
        super().__init__(self.message)