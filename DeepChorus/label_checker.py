# run this to make sure label values were saved and correlate to feature files properly 
# the Deepchorus model will fail otherwise. 
import joblib

def check_labels():
    labels = joblib.load('label_name.joblib')
    features = joblib.load('feature_name.joblib')
    for key in labels:
        print(f"key: {key}, labels[key]: {labels[key]}")
        if key not in features:
            raise KeyError(f"The following key exists in labels but not features: {key}")

if __name__ == "__main__":
    check_labels()
