from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_model(file_name="model.pkl"):
    

    iris = load_iris()
    X = iris.data
    y = iris.target

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(file_name, "wb") as f:
        pickle.dump(model, f)

    

# Run only when file executed directly
if __name__ == "__main__":
    train_and_save_model()