import pandas as pd
import random
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# -------- Helper Functions -------- #

def loading_animation(message="Processing"):
    """Show a simple loading animation"""
    print(Fore.CYAN + message, end="")
    for _ in range(3):
        sys.stdout.flush()
        time.sleep(0.5)
        print(".", end="")
    print("\n")

def sentiment_analysis(text):
    """Analyze sentiment polarity of text"""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.25:
        return "Positive"
    elif polarity < -0.25:
        return "Negative"
    else:
        return "Neutral"

def recommend_movies(dataset, genre_choice, mood, min_rating=0, num_recommendations=5):
    """Recommend movies based on genre, mood, and rating"""
    
    # Filter by genre
    filtered = dataset[dataset['Genre'].str.contains(genre_choice, case=False, na=False)]
    
    # Filter by rating
    filtered = filtered[filtered['IMDB_Rating'] >= min_rating]
    
    if filtered.empty:
        print(Fore.RED + "❌ No movies found with those filters.")
        return []
    
    # Shuffle for randomness
    filtered = filtered.sample(frac=1).reset_index(drop=True)
    
    recommendations = []
    
    for _, row in filtered.iterrows():
        overview = str(row['Overview'])
        sentiment = sentiment_analysis(overview)
        
        # Mood alignment check
        if (mood == "positive" and sentiment == "Positive") or \
           (mood == "negative" and sentiment == "Negative") or \
           (mood == "neutral" and sentiment == "Neutral"):
            recommendations.append(row)
        
        if len(recommendations) >= num_recommendations:
            break
    
    return recommendations

# -------- Main Program -------- #

def main():
    print(Fore.GREEN + "🎬 Welcome to the Mood-Based Movie Recommendation System! 🎥")
    
    try:
        # Load dataset
        dataset = pd.read_csv("imdb_top_1000.csv")
    except FileNotFoundError:
        print(Fore.RED + "❌ Dataset file not found! Please place 'imdb_top_1000.csv' in the same folder.")
        return
    
    # Fill missing values
    dataset.fillna("", inplace=True)
    
    # Build list of unique genres
    all_genres = set()
    for genres in dataset['Genre']:
        for g in str(genres).split(","):
            all_genres.add(g.strip())
    genre_list = sorted(all_genres)
    
    while True:
        print(Fore.CYAN + "\nAvailable Genres:")
        print(", ".join(genre_list))
        
        genre_choice = input(Fore.YELLOW + "\n👉 Enter a genre you like: ").strip()
        if genre_choice not in genre_list:
            print(Fore.RED + "❌ Invalid genre choice. Try again.")
            continue
        
        mood = input(Fore.MAGENTA + "👉 Enter your current mood (positive/negative/neutral): ").strip().lower()
        if mood not in ["positive", "negative", "neutral"]:
            print(Fore.RED + "❌ Invalid mood choice. Try again.")
            continue
        
        try:
            min_rating = float(input(Fore.BLUE + "👉 Enter minimum IMDB rating (0-10): ").strip())
        except ValueError:
            min_rating = 0
        
        loading_animation("🔍 Finding movies for you")
        
        recommendations = recommend_movies(dataset, genre_choice, mood, min_rating)
        
        if not recommendations:
            print(Fore.RED + "❌ Sorry, no recommendations found. Try different filters.")
        else:
            print(Fore.GREEN + f"\n✅ Here are your {len(recommendations)} recommended movies:")
            for idx, movie in enumerate(recommendations, start=1):
                print(f"{Fore.CYAN}{idx}. {movie['Series_Title']} "
                      f"({movie['Released_Year']}) - Rating: {movie['IMDB_Rating']}")
                print(f"{Fore.YELLOW}   🎬 Overview: {movie['Overview']}\n")
        
        again = input(Fore.CYAN + "🔁 Do you want to try again? (yes/no): ").strip().lower()
        if again != "yes":
            print(Fore.BLUE + "👋 Thanks for using the Movie Recommendation System. Enjoy your watch!")
            break

# Run program
if __name__ == "__main__":
    main()
