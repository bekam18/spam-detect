# Spam Detection with Naive Bayes

This project is a machine learning-based spam detection system that classifies messages as spam or not spam (ham).
The implementation utilizes Python, the Naive Bayes algorithm, and the `sklearn` library. 
It also highlights spam-indicative words and provides a probability score for the prediction.

## Features

- **Spam Detection**: Classifies emails or text messages as spam or ham.
- **Highlight Key Spam Words**: Identifies words in a message that contribute to it being classified as spam.
- **Probability Score**: Provides a numerical score indicating the likelihood of the message being spam.
- **Save and Load Model**: Trained models and vectorizers can be saved and loaded for future predictions.

## Tools and Libraries Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and utilities
- **Joblib**: Saving and loading models

## Dataset

The project uses a CSV dataset (`emails.csv`) with the following structure:
- `text`: The email or message content
- `spam`: A binary label indicating whether the message is spam (1) or not (0)

Ensure your dataset is available in the project directory.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spam-detection-naive-bayes.git
    cd spam-detection-naive-bayes
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure your dataset (`emails.csv`) is in the project directory.

## Usage

### Train and Evaluate the Model

1. Run the script to train the Naive Bayes model and evaluate its accuracy:
    ```bash
    python spam_detection.py
    ```

2. Enter a message to test its classification interactively:
    ```bash
    Enter text to predict: [Your Message Here]
    ```

### Save and Load the Model

- Save the trained model and vectorizer:
    ```bash
    python spam_detection.py
    ```

- Load the saved model and vectorizer for new predictions:
    ```bash
    python spam_detection.py
    ```

## Example Output

```text
Model Accuracy: 0.97
Enter text to predict: Win a free iPhone now!
The message is: Spam
Spam Probability: 0.85
Spam Words Highlighted: ['win', 'free', 'now']
```

## Files in the Repository

- `spam_detection.py`: Main script for training, testing, and saving the model.
- `emails.csv`: Example dataset for training and testing (not included; please add your own dataset).
- `naive_bayes_model.pkl`: Saved Naive Bayes model.
- `tfidf_vectorizer.pkl`: Saved vectorizer for text preprocessing.
- `requirements.txt`: List of required Python libraries.

## Requirements

- Python 3.8+
- Pandas
- Scikit-learn
- Joblib

## Contributing

Feel free to fork this repository and make contributions. Please submit a pull request with detailed information about the changes you propose.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Happy Coding!

