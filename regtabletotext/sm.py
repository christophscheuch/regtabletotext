import pandas as pd

def calculate_residuals_statistics(results, **options):
    """
    Calculate and return the statistics of residuals from the given regression results.
    
    Parameters:
    - results (object): A regression results object that contains a 'resid' attribute, typically from statsmodels.
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
    
    # Check if the results object has the 'resid' attribute and is from statsmodels
    if not hasattr(results, 'resid') or not 'statsmodels' in str(type(results)):
        raise ValueError("The 'results' parameter should be a single regression results object from statsmodels with a 'resid' attribute.")
    
    # Extract options or set defaults
    digits = options.get('digits', 3)
    
    # Extract residuals
    residuals = results.resid
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

def create_coefficients_table(results, **options):
    """
    Extract and format the coefficients table from regression results.
    
    This function extracts the coefficients table from the summary of regression results, 
    typically from statsmodels, and returns a formatted DataFrame with selected columns 
    and rounded values.
    
    Parameters:
    ----------
    results : object
        A regression results object that contains a 'summary' method, typically from statsmodels.
    
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

     # Check if the results object is from statsmodels
    if not 'statsmodels' in str(type(results)):
        raise ValueError("The 'results' parameter should be a single regression results object from statsmodels.")
    
    # Extract options or set defaults
    digits = options.get('digits', 3)
    
    # Extract results 
    results_data = results.summary().tables[1].data

    # Collect coefficient statistics in a data frame
    coefficients_table = (pd.DataFrame(
        results_data[1:], 
        columns=results_data[0]
      )
      .get(["", "coef", "std err", "t", "P>|t|"])
      .rename(columns={
        "coef": "Estimate",
        "std err": "Std. Error",
        "t": "t-Statistic",
        "P>|t|": "p-Value"
        },
      )
      .set_index("")
      .apply(pd.to_numeric, errors='coerce')
      .round(digits)
    )
    
    return(coefficients_table)
  
def prettify_statsmodels(results, **options):
    """
    Format and print regression results in a style similar to R's summary() output for linear models.
    
    This function takes regression results from statsmodels and prints a summary 
    that resembles the output provided by R's summary() function for linear models. The summary 
    includes the model formula, coefficients table, and other summary statistics. Optionally, 
    it can also include residuals statistics.

    Parameters:
    ----------
    results : object
        A regression results object from statsmodels that contains attributes 
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
    # Check if the results object is from statsmodels
    if not 'statsmodels' in str(type(results)):
        raise ValueError("The 'results' parameter should be a single regression results object from statsmodels.")
    
    # Extract options or set defaults
    digits = options.get('digits', 3)
    include_residuals = options.get('include_residuals', False)
    
    # Initialize the output string
    output = f"\nModel:\n{results.model.formula}\n\n"
    
    # Add residuals to the output string if required
    if include_residuals:
        output += f"Residuals:\n{calculate_residuals_statistics(results, digits=digits).to_string(index=False)}\n\n"

    # Add coefficients to the output string
    # Assuming you have a function called 'create_coefficients_table' that creates the coefficients table
    output += f"Coefficients:\n{create_coefficients_table(results, digits=digits).to_string()}\n\n"

    # Add footer with additional statistics to the output string
    output += (
        f"Summary statistics:\n"
        f"- Number of observations: {results.nobs:.0f}\n"
        f"- Residual standard error: {results.mse_resid**0.5:.{digits}f}\n"
        f"- Multiple R-squared: {results.rsquared:.{digits}f}, Adjusted R-squared: {results.rsquared_adj:.{digits}f}\n"
        f"- F-statistic: {results.fvalue:.{digits}f} on {results.df_model:.0f} and {results.df_resid:.0f} DF, p-value: {results.f_pvalue:.{digits}f}\n"
    )

    # Print the output string
    print(output)

