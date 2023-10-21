ALLOWED_OPTIONS = {'digits', 'include_residuals'}
TYPE_STATSMODELS = 'statsmodels.regression.linear_model.RegressionResultsWrapper'
TYPE_LINEARMODELS = 'linearmodels.panel.results.PanelEffectsResults'
SUPPORTED_MODELS = {TYPE_STATSMODELS, TYPE_LINEARMODELS}

import pandas as pd

def is_result_type_valid(result):
    result_type = type(result).__module__ + "." + type(result).__name__
    return(result_type in SUPPORTED_MODELS)

def is_result_type_statsmodels(result):
    result_type = type(result).__module__ + "." + type(result).__name__
    return(TYPE_STATSMODELS in result_type)

def is_result_type_linearmodels(result):
    result_type = type(result).__module__ + "." + type(result).__name__
    return(TYPE_LINEARMODELS in result_type)

def calculate_residuals_statistics(result, **options):
    """
    Calculate and return the statistics of residuals from the given regression result.
    
    Parameters:
    - result (object): A regression result object that contains a 'resid' attribute, typically from statsmodels.
    - options (dict): Optional parameters for the function. Current options include:
        * 'digits': Number of decimal places to round the calculated statistics to.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the following statistics of residuals:
        * Min: Minimum value of residuals.
        * Q25: 25th percentile (1st quartile) of residuals.
        * Q50: Median (50th percentile) of residuals.
        * Q75: 75th percentile (3rd quartile) of residuals.
        * Max: Maximum value of residuals.
    """
    
    # Check if the result object is valid
    if not is_result_type_valid(result):
        raise ValueError("The 'result' parameter should be a single regression result object from statsmodels or linearmodels.")
    
    # Extract options or set defaults
    digits = options.get('digits', 3)
    
    # Extract residuals
    if is_result_type_statsmodels(result):
        residuals = result.resid
    
    if is_result_type_linearmodels(result):
        residuals = result.resids

    stats = {
        "Min": residuals.min(),
        "Q25": residuals.quantile(0.25),
        "Q50": residuals.median(),
        "Q75": residuals.quantile(0.75),
        "Max": residuals.max()
    }

    residuals_stats = pd.DataFrame(stats, index=[0])
    
    # Apply options
    if 'digits' in options:
        residuals_stats = residuals_stats.round(digits)
    
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
    
    # Extract options or set defaults
    digits = options.get('digits', 3)
    
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
    
    return(coefficients_table)
  
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
    
    # Extract options or set defaults
    digits = options.get('digits', 3)
    include_residuals = options.get('include_residuals', False)
    
    if is_result_type_statsmodels(result):
        # Initialize the output string
        output = f"\nModel:\n{result.model.formula}\n\n"
        
        # Add residuals to the output string if required
        if include_residuals:
            output += f"Residuals:\n{calculate_residuals_statistics(result, digits=digits).to_string(index=False)}\n\n"

        # Add coefficients to the output string
        # Assuming you have a function called 'create_coefficients_table' that creates the coefficients table
        output += f"Coefficients:\n{create_coefficients_table(result, digits=digits).to_string()}\n\n"

        # Add footer with additional statistics to the output string
        output += (
            f"Summary statistics:\n"
            f"- Number of observations: {result.nobs:.0f}\n"
            f"- Multiple R-squared: {result.rsquared:.{digits}f}, Adjusted R-squared: {result.rsquared_adj:.{digits}f}\n"
            f"- F-statistic: {result.fvalue:.{digits}f} on {result.df_model:.0f} and {result.df_resid:.0f} DF, p-value: {result.f_pvalue:.{digits}f}\n"
        )
    
    if is_result_type_linearmodels(result):
        # Initialize the output string
        output = f"\nModel:\n{result.model.formula}\n\n"
        
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
str