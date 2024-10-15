from flask import Flask, render_template, request
import pandas as pd
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