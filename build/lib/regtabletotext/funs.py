# ====================
# SECTION: Imports
# ====================
import pandas as pd
from tabulate import tabulate

# ====================
# SECTION: Constants
# ====================
ALLOWED_OPTIONS = {'digits', 'include_residuals', 'max_width'}
TYPE_STATSMODELS = 'statsmodels.regression.linear_model.RegressionResultsWrapper'
TYPE_LINEARMODELS = 'linearmodels.panel.results.PanelEffectsResults'
TYPE_ARCH_MODEL = 'arch.univariate.base.ARCHModelResult'
SUPPORTED_MODELS = {TYPE_STATSMODELS, TYPE_LINEARMODELS, TYPE_ARCH_MODEL}

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

def are_result_type_linearmodels(results):
    return all(is_result_type_linearmodels(result) for result in results)

def is_result_type_arch_model(result):
    result_type = type(result).__module__ + "." + type(result).__name__
    return(result_type in TYPE_ARCH_MODEL)

# ===================
# SECTION: Helper Functions
# ====================
def clean_model_formula(model_formula, options={'max_width': 64}):
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
    options (dict): Optional parameters for the function. Current options include:
        * 'max_width': The maximum width (number of characters) that the formula should occupy on a single line (default: 64). 

    Returns:
    -------
    str
        The cleaned and formatted model formula.
    """

    max_width = options.get('max_width')
        
    model_formula_cleaned = ' '.join(model_formula.split())

    if len(model_formula_cleaned) <= max_width:
        return model_formula_cleaned
    else:
        # Split the formula at the last '+' before max_width
        split_index = model_formula_cleaned.rfind('+', 0, max_width)
        return model_formula_cleaned[:split_index] + '\n + ' + model_formula_cleaned[split_index+1:].strip()

def calculate_residuals_statistics(residuals, options={'digits': 3}):
    """
    Calculate and return the statistics of residuals from the given regression result.
    
    Parameters:
    - residuals series, Pandas series type
    - options (dict): Optional parameters for the function. Current options include:
        * 'digits': Number of decimal places to round the calculated statistics to (default: 3).

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
   
    # Extract options or use defaults
    digits = options.get('digits')

    residuals_stats = residuals.describe().iloc[1:]
    residuals_stats.index = [i.capitalize() for i in residuals_stats.index]
    residuals_stats = pd.DataFrame(residuals_stats).T.round(digits)
    residuals_stats.index.name = None

    return residuals_stats

def truncate_coefficients_table(coefficients_table, options={'max_width': 64}):
    """
    Truncate and return coefficient table with max width.
    Parameters:
    - coefficients_table, Pandas DataFrame type
    - max_width, int number of string characters per line
    Returns:
    - pd.DataFrame: A DataFrame containing coefficients
    """
    
    max_width = options.get('max_width')

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
            coefficients_table = coefficients_table.rename(
                index={coeff_name: truncated_name})
    return coefficients_table

def create_coefficients_table(result, options={'digits': 3, 'max_width': 64}):
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
        * 'max_width': The maximum width (number of characters) that the formula should occupy on a single line (default: 64). 

    Returns:
    -------
    list of
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
    digits = options.get('digits')
    max_width = options.get('max_width')

    # Initialize output list
    coefficients_tables_list = []
    
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

    if is_result_type_arch_model(result):
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

        # Extract result
        result_data_vola = result.summary().tables[2].data

        # Collect coefficient statistics in a data frame
        coefficients_table_vola = (
            pd.DataFrame(result_data_vola[1:], columns=result_data_vola[0])
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
        coefficients_table_vola = truncate_coefficients_table(coefficients_table_vola, options={'max_width': max_width})
        coefficients_table_vola.index.name = None

    coefficients_table = truncate_coefficients_table(coefficients_table, options={'max_width': max_width})
    coefficients_table.index.name = None
    coefficients_tables_list.append(coefficients_table)

    if is_result_type_arch_model(result):
        coefficients_tables_list.append(coefficients_table_vola)

    return coefficients_tables_list

def create_fixed_effects_table(result):
    """
    Create a table summarizing the fixed effects included in a panel model result.

    The table will show the total number of entities and time periods, but only for 
    the effects that are actually included in the model (based on result.included_effects).

    Parameters:
    - result (PanelEffectsResults): The results object obtained after fitting a panel data model using `linearmodels`.

    Returns:
    - DataFrame: A pandas DataFrame with the effect types (Entity, Time) as the index and the total counts as values.
                 Only the effects present in result.included_effects will be included in the DataFrame.
    """
    fixed_effects_table = pd.DataFrame({
        '': ['Entity', 'Time'],
        'Total': [result.entity_info.total, result.time_info.total]
    }).set_index('')
    fixed_effects_table = fixed_effects_table[fixed_effects_table.index.isin(result.included_effects)]
    fixed_effects_table['Total'] = fixed_effects_table['Total'].astype(int)
    fixed_effects_table.index.name = None

    return(fixed_effects_table)

# ===================
# SECTION: Main Functions
# ====================
def prettify_result(result, options={'digits': 3, 'include_residuals': False, 'max_width': 64}):
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
        * 'digits': Number of decimal places to round the values in the coefficients table to (default=3).
        * 'include_residuals': Whether to include residuals in the output (default=False).
        * 'max_width': The maximum width (number of characters) that the formula should occupy on a single line (default: 64). 

    Returns:
    -------
    None
        The function prints the formatted summary directly.
    """
    # Check if the result object is valid
    if not is_result_type_valid(result):
        raise ValueError("The 'result' parameter is currently not supported.")
    
    # Check if options are valid
    invalid_options = set(options.keys()) - ALLOWED_OPTIONS
    if invalid_options:
        raise ValueError(f"Invalid options provided: {', '.join(invalid_options)}")
    
    # Extract options or use defaults
    digits = options.get('digits', 3)
    include_residuals = options.get('include_residuals', False)
    max_width = options.get('max_width', 64)

    if is_result_type_statsmodels(result):
        # Initialize the output string
        if (hasattr(result.model, "formula")):
            model_formula = clean_model_formula(result.model.formula, options={'max_width': max_width})
            output = f"OLS Model:\n{model_formula}\n\n"
        else:
            output = f"OLS Model (no formula provided)\n\n"
        
        # Add residuals to the output string if required
        if include_residuals:
            residuals_statistics = calculate_residuals_statistics(result.resid, options={'digits': digits}).to_string(index=False)
            output += f"Residuals:\n{residuals_statistics}\n\n"

        # Add coefficients to the output string
        coefficients_table = create_coefficients_table(result, options={'digits': digits, 'max_width': max_width})[0].to_string()
        output += f"Coefficients:\n{coefficients_table}\n\n"

        # Add footer with additional statistics to the output string
        output += (
            f"Summary statistics:\n"
            f"- Number of observations: {result.nobs:,.0f}\n"
            f"- R-squared: {result.rsquared:.{digits}f}, Adjusted R-squared: {result.rsquared_adj:.{digits}f}\n"
            f"- F-statistic: {result.fvalue:,.{digits}f} on {result.df_model:.0f} and {result.df_resid:.0f} DF, p-value: {result.f_pvalue:.{digits}f}\n"
        )
    
    if is_result_type_linearmodels(result):
        # Initialize the output string
        output = f"Panel OLS Model:\n{clean_model_formula(result.model.formula)}\n\n"
        
        # Add covariance type
        output += f"Covariance Type: {result._cov_type}\n\n"

        # Add residuals to the output string if required
        if include_residuals:
            # Assuming you have a function called 'calculate_residuals_statistics' for linearmodels
            residual_statistics = calculate_residuals_statistics(result.resids, options={'digits': digits}).to_string(index=False)
            output += f"Residuals:\n{residual_statistics}\n\n"

        # Add coefficients to the output string
        coefficients_table = create_coefficients_table(result, options={'digits': digits, 'max_width': max_width})[0].to_string()
        output += f"Coefficients:\n{coefficients_table}\n\n"

        # Include table with included fixed effects (if any)
        if len(result.included_effects) > 0:
            fixed_effects_table = create_fixed_effects_table(result).to_string()
            output += f"Included Fixed Effects:\n{fixed_effects_table}\n\n"            

        # Add footer with additional statistics to the output string
        output += (
            f"Summary statistics:\n"
            f"- Number of observations: {result.nobs:,.0f}\n"
            f"- R-squared (incl. FE): {result.rsquared_inclusive:.{digits}f}, Within R-squared: {result.rsquared_within:.{digits}f}\n"
            f"- F-statistic: {result.f_statistic.stat:,.{digits}f}, p-value: {result.f_statistic.pval:.{digits}f}\n"
        )

    if is_result_type_arch_model(result):
        model_name = result.summary().as_text().split('\n')[0].strip()
        output = f"\n{model_name}\n\n"
        if include_residuals:
            residuals_statistics = calculate_residuals_statistics(result.resid, options={'digits': digits}).to_string(index=False)
            output += f"Residuals:\n{residuals_statistics}\n\n"

        # Add coefficients to the output string
        output += f"Mean Coefficients:\n{create_coefficients_table(result, options={'digits': digits, 'max_width': max_width})[0].to_string()}\n\n"
        output += f"Coefficients for {str(result.model.volatility)}:\n{create_coefficients_table(result, options={'digits': digits, 'max_width': max_width})[1].to_string()}\n\n"

        # Add footer with additional statistics to the output string
        output += (
            f"Summary statistics:\n"
            f"- Number of observations: {result.nobs:,.0f}\n"
            f"- Distribution: {str(result.model.distribution)}\n"
            f"- Multiple R-squared: {result.rsquared:.{digits}f}, Adjusted R-squared: {result.rsquared_adj:.{digits}f}\n"
            f"- BIC: {result.bic:,.{digits}f},  AIC: {result.aic:,.{digits}f}\n"
        )

    # Print the output string
    print(output)

def prettify_results(results, options={'digits': 3}):
    """
    Formats and prints a summary table of regression results from a list of 'linearmodels.panel.results.PanelEffectsResults' objects.
    
    The function prints a table that includes the dependent variables, estimated coefficients with t-statistics, fixed effects, 
    variance-covariance (VCOV) type, number of observations, inclusive R-squared, and within R-squared. Coefficients and t-statistics 
    are rounded to the specified number of digits. 
    
    Parameters:
        results (list): A list of 'linearmodels.panel.results.PanelEffectsResults' objects containing regression results.
        options (dict, optional): A dictionary containing formatting options. Currently supports:
            - 'digits' (int): The number of decimal places to round the coefficients and t-statistics. Default is 3.
            
    Raises:
        ValueError: If 'results' is not a list of 'linearmodels.panel.results.PanelEffectsResults' objects.
        
    Returns:
        None: Prints the formatted summary table to the console.
    """

    if not are_result_type_linearmodels(results):
        raise ValueError("Only list of results of type 'linearmodels.panel.results.PanelEffectsResults' are supported.")
    
    # Extract options or use defaults
    digits = options.get('digits', 3)

    # Dependent variables
    dependent_vars = ["".join(result.model.dependent.vars)  for result in results]
    dependent_vars.insert(0, "Outcome")

    # Coefficients with t-stats in parentheses 
    dfs = []
    for i, result in enumerate(results):
        table = create_coefficients_table(result)[0]
        coefs = table.get("Estimate").round(digits)
        tstats = table.get("t-Statistic").round(2)
        combined = coefs.astype(str) + " (" + tstats.astype(str) + ")"
        dfs.append(combined.to_frame(name=f'Estimate_{i+1}'))

    merged_df = pd.concat(dfs, axis=1, join='outer')

    coefficients = merged_df.reset_index().fillna('').astype(str).values.tolist()

    ## Fixed effects (if any)
    included_effects = [', '.join(result.included_effects)  for result in results]
    included_effects.insert(0, "Fixed effects")

    ## VCOV type
    cov_type = [result._cov_type for result in results]
    cov_type.insert(0, "VCOV type")

    # Observations
    nobs = [f'{result.nobs:,.0f}' for result in results]
    nobs.insert(0, "Observations")

    # R2
    rsquared_inclusive = [f'{result.rsquared_inclusive:,.{digits}f}' for result in results]
    rsquared_inclusive.insert(0, "R2 (incl. FE)")

    # Within R-squared: 
    rsquared_within = [f'{result.rsquared_within:,.{digits}f}' for result in results]
    rsquared_within.insert(0, "Within R2")

    # Assemble table
    table = []
    table.append(dependent_vars)
    table.append("")
    for sublist in coefficients:
        table.append(sublist)
    table.append("")
    table.append(included_effects)
    table.append(cov_type)
    table.append(nobs)
    table.append(rsquared_inclusive)
    table.append(rsquared_within)

    output = tabulate(table, tablefmt="plain", colalign=("left",) + ("center",) * (len(results)))

    print(output)