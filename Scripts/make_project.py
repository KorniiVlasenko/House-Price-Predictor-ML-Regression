# This script combines all other scripts so that you can build a project by executing just one file


# Execute preprocessing script
import preprocessing as pp
pp.full_preprocessing()

# Execute models_tuning_and_evaluation script
import models_tuning_and_evaluation as mod_te
mod_te.save_tuned_models()

# Execute models_prediction_on_different_data script
import models_prediction_on_different_data as models_preds
models_preds.get_all_predictions()

# Execute best_model_fit_predict script
import best_model_fit_predict as best
best.make_best_prediction()



