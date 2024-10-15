from flask import Flask, render_template, request
import pandas as pd
<<<<<<< HEAD
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Load the recipe dataset
    data = pd.read_csv('recipes.csv')

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    ingredient_matrix = vectorizer.fit_transform(data['ingredients'])

    logging.info("Data loaded and TF-IDF matrix created successfully")
except Exception as e:
    logging.error(f"Error initializing data: {str(e)}")
    raise


@app.route('/', methods=['GET', 'POST'])
def recommend():
    try:
        ingredients_list = data['ingredients'].str.split(', ').explode().unique().tolist()

        selected_ingredients = request.form.getlist('ingredients')
        selected_dietary_restriction = request.form.get('dietary_restriction', 'no-selection')
        selected_pregnancy_restriction = request.form.get('pregnancy_restriction', 'no-selection')
        sort_option = request.form.get('sort_option', 'relevance')

        if request.method == 'POST':
            # Start with all recipes
            recommendations = data.copy()

            # Calculate similarity scores using ML (TF-IDF and cosine similarity)
            if selected_ingredients:
                user_vector = vectorizer.transform([' '.join(selected_ingredients)])
                similarity_scores = cosine_similarity(user_vector, ingredient_matrix).flatten()
                recommendations['similarity'] = similarity_scores
                recommendations = recommendations[recommendations['similarity'] > 0]
            else:
                recommendations['similarity'] = 1  # Set default similarity if no ingredients selected

            # Apply dietary restrictions
            if selected_dietary_restriction != 'no-selection':
                recommendations = recommendations[
                    recommendations['dietary_restrictions'].str.contains(selected_dietary_restriction, case=False,
                                                                         na=False)
                ]

            # Apply pregnancy/newborn restrictions
            if selected_pregnancy_restriction == 'pregnant woman':
                recommendations = recommendations[recommendations['pregnant_women_safe'] == 'Yes']
            elif selected_pregnancy_restriction == 'newborn babies':
                recommendations = recommendations[recommendations['newborn_safe'] == 'Yes']

            # Convert 'time_to_prepare' to numeric
            recommendations['time_to_prepare'] = pd.to_numeric(
                recommendations['time_to_prepare'].str.replace(' minutes', ''), errors='coerce')

            # Sort recommendations
            if sort_option == 'time_to_prepare':
                recommendations = recommendations.sort_values(by='time_to_prepare')
            elif sort_option == 'similarity':
                recommendations = recommendations.sort_values(by='similarity', ascending=False)
            else:  # relevance (most relevant)
                # Normalize similarity and time_to_prepare
                recommendations['norm_similarity'] = (recommendations['similarity'] - recommendations[
                    'similarity'].min()) / (recommendations['similarity'].max() - recommendations['similarity'].min())
                recommendations['norm_time'] = (recommendations['time_to_prepare'] - recommendations[
                    'time_to_prepare'].min()) / (recommendations['time_to_prepare'].max() - recommendations[
                    'time_to_prepare'].min())

                # Calculate relevance score (higher similarity and lower time are more relevant)
                recommendations['relevance_score'] = recommendations['norm_similarity'] - 0.5 * recommendations[
                    'norm_time']

                recommendations = recommendations.sort_values(by='relevance_score', ascending=False)

            # Convert DataFrame to a list of dictionaries
            recommendations_list = recommendations.to_dict(orient='records')

            logging.info(
                f"Generated recommendations for {len(selected_ingredients)} ingredients, {selected_dietary_restriction} diet, {selected_pregnancy_restriction} restriction, sorted by {sort_option}")

            return render_template('index.html', recipes=recommendations_list, ingredients=ingredients_list,
                                   selected_ingredients=selected_ingredients,
                                   selected_dietary_restriction=selected_dietary_restriction,
                                   selected_pregnancy_restriction=selected_pregnancy_restriction,
                                   sort_option=sort_option)

        return render_template('index.html', recipes=None, ingredients=ingredients_list)

    except Exception as e:
        logging.error(f"Error in recommend function: {str(e)}")
        return render_template('error.html', error_message="An error occurred while processing your request.")


if __name__ == '__main__':
    app.run(debug=True)
=======
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the recipe dataset
data = pd.read_csv('recipes.csv')

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['ingredients'])

# Prepare the data for the ML model
y = data[['pregnant_women_safe', 'newborn_safe']].apply(lambda x: 1 if x['pregnant_women_safe'] == 'Yes' else 0, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def recommend():
    ingredients_list = data['ingredients'].str.split(', ').explode().unique().tolist()

    selected_ingredients = request.form.getlist('ingredients')
    selected_dietary_restriction = request.form.get('dietary_restriction', 'no-selection')
    selected_pregnancy_restriction = request.form.get('pregnancy_restriction', 'no-selection')
    sort_option = request.form.get('sort_option', 'relevance')

    if request.method == 'POST':
        # Check if no ingredients are selected
        if not selected_ingredients:
            recommendations = data.copy()
        else:
            recommendations = data

            # Filter based on selected ingredients using cosine similarity
            selected_ingredient_vector = vectorizer.transform([' '.join(selected_ingredients)])
            cosine_similarities = cosine_similarity(selected_ingredient_vector, X).flatten()

            # Add cosine similarity scores to recommendations
            recommendations['similarity'] = cosine_similarities
            recommendations = recommendations[recommendations['similarity'] > 0]  # Filter out recipes with no similarity

        # Apply dietary restrictions
        if selected_dietary_restriction != 'no-selection':
            recommendations = recommendations[
                recommendations['dietary_restrictions'].str.contains(selected_dietary_restriction, case=False, na=False)
            ]

        # Use ML model to predict safety for pregnant women
        if selected_pregnancy_restriction == 'pregnant woman':
            preg_safe_indices = model.predict(vectorizer.transform(recommendations['ingredients']))
            recommendations = recommendations[preg_safe_indices == 1]
        elif selected_pregnancy_restriction == 'newborn babies':
            newborn_safe_indices = model.predict(vectorizer.transform(recommendations['ingredients']))
            recommendations = recommendations[newborn_safe_indices == 1]

        # Sort by time, relevance, or similarity
        if sort_option == 'time_to_prepare':
            recommendations['time_to_prepare'] = pd.to_numeric(recommendations['time_to_prepare'].str.replace(' minutes', ''), errors='coerce')
            recommendations = recommendations.sort_values(by='time_to_prepare')
        elif sort_option == 'similarity':
            recommendations = recommendations.sort_values(by='similarity', ascending=False)

        # Convert DataFrame to a list of dictionaries
        recommendations_list = recommendations.to_dict(orient='records')

        return render_template('index.html', recipes=recommendations_list, ingredients=ingredients_list,
                               selected_ingredients=selected_ingredients,
                               selected_dietary_restriction=selected_dietary_restriction,
                               selected_pregnancy_restriction=selected_pregnancy_restriction,
                               sort_option=sort_option)

    return render_template('index.html', recipes=None, ingredients=ingredients_list)

if __name__ == '__main__':
    app.run(debug=True)


'''
NLP (Natural Language Processing)
TF-IDF Vectorization: You employed the TfidfVectorizer to convert the ingredients into a numerical format that captures the importance of each ingredient in relation to others. This is a common NLP technique used to represent text data in a form that machine learning algorithms can understand.
ML (Machine Learning)

Logistic Regression Model: You trained a LogisticRegression model to predict whether a recipe is safe for pregnant women or newborn babies based on the ingredients.
You prepared the data for the model using the features derived from the TF-IDF vectorization and defined the target variable based on the pregnant_women_safe and newborn_safe columns in your dataset.
You split the dataset into training and testing sets to train the model and make predictions.

Summary

Cosine Similarity: This is used to find the similarity between the selected ingredients and the recipe ingredients, which can be considered a part of information retrieval, often integrated with NLP.

In summary, you effectively integrated both NLP and ML techniques to create a recommendation system for recipes based on user input! If you need help with further enhancements or explanations, feel free to ask!
'''
>>>>>>> 6799016a1c901e691f8ce7f0aeef2f65855d664b
