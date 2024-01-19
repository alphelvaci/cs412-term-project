## CS412 Course Project - Can we predict HW grades from ChatGPT interactions?

*by Emir Balkan, Alp Helvacı, Ilgın Arat*

### Overview: 
This project aims to evaluate homework submissions based on various features extracted from the responses. The features include question matching, length and count analysis, also vectorization of prompts and answers. The final goal is to build a machine learning model that predicts the grades of the homework submissions.

### Project Structure: 
The project is organized into several components:

### 1.	Preprocessing: 
⁠First, We extract the HTML data of prompts of students and answers of the Chatgpt history. Then, we matched them as prompt-answer pairs and stored it in json file. Lastly, we cleaned it from the punctuations, unnecessary spaces, and html tags. At the end, we had only words and numbers as strings.  

### 2.	Feature Creation Functions: 
These functions define the procedures for creating different features used for evaluating homework submissions. The functions include: <br />

•	question_matching: Matches prompts with predefined keywords for each question and counts the number of matches for each question. The keywords are as following: <br />
      keywords = { <br />
        'q0': set(['load', 'dataset', 'csv', 'file']), <br />
        'q1': set(['shape', 'summary', 'head', 'map', 'missing', 'label']), <br />
        'q2': set(['shuffle', 'seperate', 'split', 'training', '80', '20']), <br />
        'q3': set(['correlation', 'feature', 'selection', 'hypothetical']), <br />
        'q4': set(['hyperparameter', 'tune', 'gridsearchcv']), <br />
        'q5': set(['retrain', 'hyperparameter', 'decision', 'tree', 'plot']), <br />
        'q6': set(['predict', 'classification', 'accuracy', 'confusion', 'matrix']), <br />
        'q7': set(['information', 'gain', 'entropy', 'formula']) <br />
    } <br />
•	length_and_count: Computes the average lengths of prompts and answers, also counts the number of number-answer pairs. <br />
•	vectorized_prompts and vectorized_answers: Vectorizes prompts and answers using pre-trained word embeddings from Word2Vec. It uses 100-dimensional vectors to express each Word.

### 3. Row Processing: 
The our_super_great_row_processor function combines the feature creation functions to process each row in the dataset.

### 4. Data Loading: 
The project loads raw data from a JSON file (raw_data.json) containing prompt-answer pairs and their corresponding grades. It also loads pre-trained word embeddings using the Gensim library.

### 5. Data Preparation: 
The script prepares the data for machine learning by creating a data frame with columns for prompt vectors, answer vectors, question matches, and other features. It also reads grades from a CSV file (scores.csv) and associates them with the corresponding homework submissions.

### 6. Model Building: 
The project utilizes a neural network model built using the Keras library. The model consists of multiple dense layers with ReLU activation functions and an output layer that uses the linear activation function. To ensure that all features fed into the model contribute equally to the learning process, we used data scaling. All the features created are inserted into the neural network aiming the model will adjust the features (weights, etc.) to get the most accurate predictions.

### 7. Model Training: 
The model is trained on the prepared data using the Adam optimizer and mean squared error loss. The training process involves several epochs with batch training.

### 7. Outcomes: 
51.08 MSE is reached with the Neural Network Model trained.

### Instructions: 
1.	Ensure you have the required dependencies installed, including Gensim, NumPy, Pandas, and Keras.
2.	Download the necessary data files (dataset and scores.csv) and place them in the appropriate directories.
3.	Run the script to process the data, build the model, and train it on the homework submissions.
4.	Analyze the model summary and training results to evaluate its performance.
5.	Make predictions on new data or fine-tune the model as needed.
