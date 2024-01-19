Overview
This project aims to evaluate homework submissions based on various features extracted from the responses. The features include question matching, length and count analysis, also vectorization of prompts and answers. The final goal is to build a machine learning model that predicts the grades of the homework submissions.

Project Structure
The project is organized into several components:

1.	Preprocessing
⁠First, We extract the HTML data of prompts of students and answers of the Chatgpt history. Then, we matched them as prompt-answer pairs and stored it in json file. Lastly, we cleaned it from the punctuations, unnecessary spaces, and html tags. At the end, we had only words and numbers as strings.  

2.	Feature Creation Functions
These functions define the procedures for creating different features used for evaluating homework submissions. The functions include:
•	question_matching: Matches prompts with predefined keywords for each question and counts the number of matches for each question.
•	length_and_count: Computes the average lengths of prompts and answers, also counts the number of number-answer pairs.
•	vectorized_prompts and vectorized_answers: Vectorizes prompts and answers using pre-trained word embeddings from Word2Vec. It uses 100 dimensional vectors to express each Word.

2. Row Processing
The our_super_great_row_processor function combines the feature creation functions to process each row in the dataset.

3. Data Loading
The project loads raw data from a JSON file (raw_data.json) containing prompt-answer pairs and their corresponding grades. It also loads pre-trained word embeddings using the Gensim library.

4. Data Preparation
The script prepares the data for machine learning by creating a DataFrame with columns for prompt vectors, answer vectors, question matches, and other features. It also reads grades from a CSV file (scores.csv) and associates them with the corresponding homework submissions.

5. Model Building
The project utilizes a neural network model built using the Keras library. The model consists of multiple Dense layers with ReLU activation functions. All the features created will be fed into the neural network aiming it will adjust the features to get the most accurate predictions.

6. Model Training
The model is trained on the prepared data using the Adam optimizer and mean squared error loss. The training process involves several epochs with batch training.


Instructions
1.	Ensure you have the required dependencies installed, including Gensim, NumPy, Pandas, and Keras.
2.	Download the necessary data files (dataset and scores.csv) and place them in the appropriate directories.
3.	Run the script to process the data, build the model, and train it on the homework submissions.
4.	Analyze the model summary and training results to evaluate its performance.
5.	Make predictions on new data or fine-tune the model as needed.
