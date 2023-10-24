# ====================
# SECTION: Imports
# ====================
import pandas as pd

# ====================
# SECTION: Constants
# ====================
ALLOWED_OPTIONS = {'digits', 'include_residuals', 'max_width'}
DEFAULT_DIGITS = 3
DEFAULT_INCLUDE_RESIDUALS = False
DEFAULT_MAX_WIDTH = 64
TYPE_STATSMODELS = 'statsmodels.regression.linear_model.RegressionResultsWrapper'
TYPE_LINEARMODELS = 'linearmodels.panel.results.PanelEffectsResults'
SUPPORTED_MODELS = {TYPE_STATSMODELS, TYPE_LINEARMODELS}

# ===================
# SECTION: Validation Functions
# ====================
def is_result_type_valid(result):
    result_type = type(result).__module__ + "." + type(result).__name__
    return(result_type in SUPPORTED_MODELS)

def is_result_type_statsmodels(result):
    result_type = type(result).__module__ + "." + type(result).__name__
    return(result_type in TYPE_STATSMODELS)

def is_result_type_linearmodels(result):
    result_type = type(result).__module__ + "." + type(result).__name__
    return(result_type in TYPE_LINEARMODELS)

# ===================
# SECTION: Helper Functions
# ====================
def clean_model_formula(model_formula, max_width=DEFAULT_MAX_WIDTH):
    """
    Cleans and formats a given model formula to fit within a specified width.

    This function takes a model formula as input, removes any extra spaces, and ensures that the 
    formula does not exceed the specified maximum width (`max_width`). If the formula exceeds the 
    `max_width`, it splits the formula at the last '+' sign before the `max_width` and continues 
    the formula on the next line.

    Parameters:
    ----------
    model_formula : str
        The model formula to be cleaned and formatted. Typically, this is a string representation 
        of a regression model formula, e.g., "y ~ x1 + x2 + x3".

    max_width : int, optional
        The maximum width (number of characters) that the formula should occupy on a single line. 
        If the formula exceeds this width, it will be split and continued on the next line. 
        Default is set by `DEFAULT_MAX_WIDTH`.

    Returns:
    -------
    str
        The cleaned and formatted model formula.

    Examples:
    --------
    >>> clean_model_formula("y ~ x1 + x2 + x3", max_width=10)
    "y ~ x1 +\n x2 + x3"
    """
        
    model_formula_cleaned = ' '.join(model_formula.split())

    if len(model_formula_cleaned) <= max_width:
        return model_formula_cleaned
    else:
        # Split the formula at the last '+' before max_width
        split_index = model_formula_cleaned.rfind('+', 0, max_width)
        return model_formula_cleaned[:split_index] + '\n + ' + model_formula_cleaned[split_index+1:].strip()

def calculate_residuals_statistics(result, **options):
    """
    Calculate and return the statistics of residuals from the given regression result.
    
    Parameters:
    - result (object): A regression result object that contains a 'resid' attribute, typically from statsmodels.
    - options (dict): Optional parameters for the function. Current options include:
        * 'digits': Number of decimal places to round the calculated statistics to.

    Returns:
    - pd.DataFrame: A DataFrame containing the following statistics of residuals:
        * Mean: Mean value of residuals.
        * Std: Standard deviation of residuals.
        * Min: Minimum value of residuals.
        * 25%: 25th percentile (1st quartile) of residuals.
        * 50%: Median (50th percentile) of residuals.
        * 75%: 75th percentile (3rd quartile) of residuals.
        * Max: Maximum value of residuals.
    """

    # Check if the result object is valid
    if not is_result_type_valid(result):
        raise ValueError("The 'result' parameter should be a single regression result object from statsmodels or linearmodels.")
    
    # Extract options or use defaults
    digits = options.get('digits', DEFAULT_DIGITS)

    # Extract residuals
    if is_result_type_statsmodels(result):
        residuals = result.resid

    if is_result_type_linearmodels(result):
        residuals = result.resids

    residuals_stats = residuals.describe().iloc[1:]
    residuals_stats.index = [i.capitalize() for i in residuals_stats.index]
    residuals_stats = pd.DataFrame(residuals_stats).T.round(digits)

    return residuals_stats

def create_coefficients_table(result, **options):
    """
    Extract and format the coefficients table from regression result.
    
    This function extracts the coefficients table from the summary of regression result, 
    typically from statsmodels, and returns a formatted DataFrame with selected columns 
    and rounded values.
    
    Parameters:
    ----------
    result : object
        A regression result object that contains a 'summary' method, typically from statsmodels.
    
    options : dict
        Optional parameters for the function. Current options include:
        * 'digits': Number of decimal places to round the values in the coefficients table to (default=3).

    Returns:
    -------
    pd.DataFrame: 
        A DataFrame containing the following columns:
        * Estimate: Coefficient estimates.
        * Std. Error: Standard error of the coefficients.
        * t-Statistic: t-statistic values.
        * p-Value: p-values associated with the t-statistics.
    """

    # Check if the result object is valid
    if not is_result_type_valid(result):
        raise ValueError("The 'result' parameter should be a single regression result object from statsmodels or linearmodels.")
    
    # Extract options or use defaults
    digits = options.get('digits', DEFAULT_DIGITS)
    max_width = options.get('max_width', DEFAULT_MAX_WIDTH)
    
    if is_result_type_statsmodels(result):
        # Extract result 
        result_data = result.summary().tables[1].data

        # Check if 't' column is present, otherwise use 'z' column
        column_to_use = 't' if 't' in result_data[0] else 'z'
        p_value_column = 'P>|t|' if 't' in result_data[0] else 'P>|z|'

        # Collect coefficient statistics in a data frame
        coefficients_table = (
            pd.DataFrame(result_data[1:], columns=result_data[0])
            .get(["", "coef", "std err", column_to_use, p_value_column])
            .rename(columns={
                "coef": "Estimate",
                "std err": "Std. Error",
                column_to_use: "Statistic",
                p_value_column: "p-Value"
            })
            .set_index("")
            .apply(pd.to_numeric, errors='coerce')
            .round(digits)
        )

    if is_result_type_linearmodels(result):
        # Extract result 
        result_data = result.summary.tables[1].data

        # Collect coefficient statistics in a data frame
        coefficients_table = (pd.DataFrame(
                result_data[1:], 
                columns=result_data[0]
            )
            .get(["", "Parameter", "Std. Err.", "T-stat", "P-value"])
            .rename(columns={
                "Parameter": "Estimate",
                "Std. Err.": "Std. Error",
                "T-stat": "t-Statistic",
                "P-value": "p-Value"
                },
            )
            .set_index("")
            .apply(pd.to_numeric, errors='coerce')
            .round(digits)
        )

    # Truncate coefficient names if they exceed max_width
    for index, row in coefficients_table.iterrows():
        coeff_name = index
        values_str = ' '.join(map(str, row.values))
        
        # Check if the combined width exceeds max_width
        if len(coeff_name) + len(values_str) + 20 > max_width: 
            # Calculate how much to truncate the coefficient name
            trunc_length = max_width - len(values_str) - 23 
            truncated_name = coeff_name[:trunc_length] + "..."
            
            # Update the index in the DataFrame
            coefficients_table = coefficients_table.rename(index={coeff_name: truncated_name})
    
    return(coefficients_table)

# ===================
# SECTION: Main Functions
# ====================
def prettify_result(result, **options):
    """
    Format and print regression result in a style similar to R's summary() output for linear models.
    
    This function takes regression result from statsmodels and prints a summary 
    that resembles the output provided by R's summary() function for linear models. The summary 
    includes the model formula, coefficients table, and other summary statistics. Optionally, 
    it can also include residuals statistics.

    Parameters:
    ----------
    result : object
        A regression result object from statsmodels that contains attributes 
        like 'model', 'nobs', 'mse_resid', 'rsquared', 'rsquared_adj', 'fvalue', 'df_model', 
        'df_resid', and 'f_pvalue'.

    options : dict
        Optional parameters for the function. Current options include:
        * 'digits': Number of decimal places to round the values in the summary to (default=3).
        * 'include_residuals': Whether to include residuals in the output (default=False).

    Returns:
    -------
    None
        The function prints the formatted summary directly.
    """
    # Check if the result object is valid
    if not is_result_type_valid(result):
        raise ValueError("The 'result' parameter should be a single regression result object from statsmodels or linearmodels.")
    
    # Check if options are valid
    invalid_options = set(options.keys()) - ALLOWED_OPTIONS
    if invalid_options:
        raise ValueError(f"Invalid options provided: {', '.join(invalid_options)}")
    
    # Extract options or use defaults
    digits = options.get('digits', DEFAULT_DIGITS)
    include_residuals = options.get('include_residuals', DEFAULT_INCLUDE_RESIDUALS)
    max_width = options.get('max_width', DEFAULT_MAX_WIDTH)

    if is_result_type_statsmodels(result):
        # Initialize the output string
        output = f"OLS Model:\n{clean_model_formula(result.model.formula, max_width=max_width)}\n\n"
        
        # Add residuals to the output string if required
        if include_residuals:
            output += f"Residuals:\n{calculate_residuals_statistics(result, digits=digits).to_string(index=False)}\n\n"

        # Add coefficients to the output string
        # Assuming you have a function called 'create_coefficients_table' that creates the coefficients table
        output += f"Coefficients:\n{create_coefficients_table(result, digits=digits, max_width=max_width).to_string()}\n\n"

        # Add footer with additional statistics to the output string
        output += (
            f"Summary statistics:\n"
            f"- Number of observations: {result.nobs:.0f}\n"
            f"- Multiple R-squared: {result.rsquared:.{digits}f}, Adjusted R-squared: {result.rsquared_adj:.{digits}f}\n"
            f"- F-statistic: {result.fvalue:.{digits}f} on {result.df_model:.0f} and {result.df_resid:.0f} DF, p-value: {result.f_pvalue:.{digits}f}\n"
        )
    
    if is_result_type_linearmodels(result):
        # Initialize the output string
        output = f"Panel OLS Model:\n{clean_model_formula(result.model.formula)}\n\n"
        
        # Add residuals to the output string if required
        if include_residuals:
            # Assuming you have a function called 'calculate_residuals_statistics' for linearmodels
            output += f"Residuals:\n{calculate_residuals_statistics(result, digits=digits).to_string(index=False)}\n\n"

        # Add coefficients to the output string
        # Assuming you have a function called 'create_coefficients_table' for linearmodels
        output += f"Coefficients:\n{create_coefficients_table(result, digits=digits).to_string()}\n\n"

        # Add footer with additional statistics to the output string
        output += (
            f"Summary statistics:\n"
            f"- Number of observations: {result.nobs:.0f}\n"
            f"- Overall R-squared: {result.rsquared_overall:.{digits}f}, Within R-squared: {result.rsquared_within:.{digits}f}\n"
            f"- F-statistic: {result.f_statistic.stat:.{digits}f}, p-value: {result.f_statistic.pval:.{digits}f}\n"
        )

    # Print the output string
    print(output)
