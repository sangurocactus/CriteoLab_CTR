# CriteoLab Click-Through-Rate

> Course: w261 Final Project (Criteo Click Through Rate Challenge)

> Team Number:19

> Team Members: Steve Dille, Naga Akkineni, Joanna Yu, Pauline Wang

> Fall 2019: section 1, 3, 4

## Our final submission comprised of the following components:
* Jypter notebook - main Jupyter notebook of our project.
* .py files - Python files used for GCP submission to run the models in the cloud.
> 1. **'steve_ctr_DT_full.py'** - GCP job for Decision Tree base model (Model I) on the full training set.
> 2. **'final_proj_GCP_RF_SSI2.py'** - GCP job for Random Forest base model (Model II) on the toy dataset.
> 3. **'steve_ctr_RF_full.py'** - GCP job for Random Forest base model (Model II) on the full training set.
> 4. **'steve_ctr_ '** - GCP job for Random Forest model with preprocessing, meaning scaled integer features and string indexed categoricl features (Model III) on the full training set.
> 5. **'steve_crt_RF_full.py'** - GCP job for Random Forest model with preprocessing listed in Model III plus Gradient Boosting (Model IV) on the full training set.
* Pickle files - Pickle files for the Pandas tables that we created for pretty printing:
> 1. **'summary.pkl'** stores the summary statistics table for the training dataset.
> 2. **'correlation.pkl'** stores all the pairwise correlation values for all 39 features and the label for the training dataset.
> 3. **'correlation_subset.pkl'** takes the table from 'correlation.pkl' and filter out entries with correlation values > 0.5.
> 4. **'DT_base_PD.pkl'** stores the summary metrics table for the Decision Tree base model with varied MaxDepth to compare model performance.
> 5. **'toyDF_PD.pkl'** stores the Pandas dataframe for the toy dataset used for EDA. The toy dataset is 1.5% of the training set.