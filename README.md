# Causal-Inference-Experiments
Fault Localization using Causal Inference and Machine Learning(Random Forests) trials on toy Python programs. Numpy,Pandas and SciKit Learn is used.
#Requirements
- Numpy,Pandas and SciKit Learn libs
- Requires Gated Single Assignment Form Translated program- Use Python GSA_Gen.
# Main components
- programX_runner.py - Runner script to make multiple runs to predict the buggy variable in a program.
- record_locals() - Call for each function's variable values to be recorded. 
- predict_causal_risk_list() and suspicious_ranking() - Where the magic happens, we use the recorded variables and the test outcomes, and predict counterfactuals to find the faulty variable.


