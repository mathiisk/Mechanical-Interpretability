# Mechanical-Interpretability
There are folders: LLM and TSC


# 1. LLM
This contains two methods I intially played around with on LLMs. 
I decided to use GPT2-XL as the paper I used for inspiration (https://arxiv.org/pdf/2202.05262 ROME used it with great results)
In observational method, I simply visualized the Avg Difference between Attention Heads for two similar prompts. ("The Capital of France is" vs "The Capital of Italy is"). The result is in the Plots folder
Then I used the insights these plots gave to perform Activation Patching (under Intervention folder). I tried different approaches to see what works but in general I follow the sturcture outlined in https://arxiv.org/pdf/2404.15255 which is as follows:
"A simple activation patching procedure typically looks like this (copied from the paper):
1. Choose two similar prompts that differ in some key fact or otherwise elicit different model
behaviour:
E.g. “The Colosseum is in” and “The Louvre is in” to vary the landmark but control
for everything else.
2. Choose which model activations to patch
E.g. MLP outputs
3. Run the model with the first prompt—the source prompt—and save its internal activations
E.g. “The Louvre is in” (source)
4. Run the model with the second prompt—the destination prompt—but overwrite the selected
internal activations with the previously saved ones (patching)
E.g. “The Colosseum is in” (destination)
5. See how the model output has changed. The outputs of this patched run are typically
somewhere between what the model would output for the un-patched first prompt or second
prompt
E.g. observe change in the output logits for "Paris" and "Rome"
6. Repeat for all activations of interest
E.g. sweep to test all MLP layers
"

Then I saved the outputs and as can be seen in the 'layersdiff.png' that layer 39 has notable differences
and patching that layer (entirly and/or head by head) changed the output. NOTE, patching other layers did not impact the output,
still the testing was not extensive.

# 2. TSC

Then after initially playing around as per suggestion I started work on TSC using Catch22. Similarly there are observation and intervention 
methods which are inspired from the previous work. Initially I did a simple test  both approaches to figure things out (same as LLMs, just playing around) and these are
in 'Intervention.py' and 'Observation.py' and I noticed that the results are interesting, thus I decided to conduct a more thoughtout approach. This is the 'AggregateIntervention.py' file.

What it does:
Implements multiple classes to structure the data processing, model training, and intervention analysis:

1. DataProcessor:

- Loads and transforms a dataset with Catch22.

- Stores the resulting DataFrame of features and class labels.

- Computes mean feature values per class (class “profiles”).

- Saves class means to Outputs/ClassMeans.csv.

2. ModelTrainer:

- Splits the data into training and test sets.

- Trains a logistic regression model on the Catch22 features.

3. InterventionAnalysis:

- Performs “patching” or “intervention” by replacing features with the average value from the other class to see how predictions change.

- Calculates how much each feature, when patched, changes the predicted probability of the originally predicted class.

- Identifies the most influential features for each test instance.

- Collects aggregate statistics: number of predictions changed vs. unchanged, average changes, etc.

- Performs statistical tests (one-sample t-tests and paired t-tests) on how patching impacts prediction probabilities, and binomial tests on how often the prediction flips.

- Saves all results into the Outputs folder (CSV files, TXT summaries, etc.).

Then for I use the results to visualize. This is done in 'PlottingHelper.py' Currently, there is the following:
Mean_Features_Class_<cls>.png – for the average feature values of each class.

1. Mean_Features_Differences.png – difference in means (for binary classification).

2. Probability_Changes_Line.png and Probability_Changes_Bar.png – to illustrate baseline vs. patched probabilities.

3. Baseline_vs_Intervention_Scatter.png – scatter of baseline probability vs. the absolute difference after intervention.

4. Prediction_Change_Ratio.png – bar chart of how many instances changed vs. did not change predictions.

5. DivergencePlot.png – diverging bar chart for changed vs. unchanged patches per feature.

I would like to go over these results in person to explain them in more detail. 

Lastly, since the goal is to determine how effective these methods are, I quickly ran SHAP on the same datasets and was pleased to see that the result is 
largerly similar to what my method achieved. I tried to resemble SHAP plot in Divergence plot. 

# HOW TO RUN

All the required dependencies are in 'requirements.txt'

        conda create -n MechInter
        conda activate MechInter
        pip install -r requirements.txt

Running LLM part might take long time (especially first time since it will download the model) so I do not recommend it for now.
As for the TSC part, it AggregateIntervention can be simply run to obtain all the results in matter of seconds and then simply running
PlottingHelper will give the plots just as quick.
