# Create a file called retrain_models.py

from train_crypto_model import CryptoPredictor

def retrain_all_models():
    """Retrain all prediction models with the latest data"""
    print("Loading CryptoPredictor...")
    predictor = CryptoPredictor()
    
    print("Loading data...")
    predictor.load_data()
    
    print("Training models with new data...")
    predictor.train_all_models()
    
    print("Evaluating models...")
    evaluation_results = predictor.evaluate_all_models()
    
    print("Model retraining completed!")
    
    # Print current predictions
    print("\nCurrent predictions:")
    for symbol in predictor.symbols:
        if symbol in predictor.data:
            prediction = predictor.make_predictions_with_threshold(symbol, confidence_threshold=0.3)
            if prediction:
                print(f"{symbol}: {prediction['direction']} with {prediction['probability']:.2f} probability " + 
                      f"(Actionable: {prediction['actionable']})")

if __name__ == "__main__":
    retrain_all_models()