import os
import time

import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def load_name_pairs(filepath="de-la-cruz.csv"):
    """
    Creates a DataFrame of Nahuatl words and their English translations,
    skipping entries with empty translations
    """
    df = pd.read_csv(filepath)

    name_pairs = df[["nahuatl", "english"]]
    name_pairs = name_pairs.dropna(subset=["english"])
    name_pairs = name_pairs.reset_index(drop=True)

    return name_pairs


def get_gpt_translation(word, client):
    """
    Gets GPT-4's translation for a Nahuatl word
    """
    try:
        prompt = f"Translate this Nahuatl term to English (respond with just the translation): {word}"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50,
        )

        translation = response.choices[0].message.content.strip()
        return translation

    except Exception as e:
        print(f"Error getting translation for {word}: {str(e)}")
        return "ERROR"


def get_deepseek_translation(word, client):
    """
    Gets DeepSeek's translation for a Nahuatl word
    """
    try:
        prompt = f"Translate this Nahuatl term to English:{word}"

        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )

        translation = response.choices[0].message.content.strip()
        return translation

    except Exception as e:
        print(f"Error getting DeepSeek translation for {word}: {str(e)}")
        return "ERROR"


def add_deepseek_translations(df):
    """
    Adds DeepSeek translations as a new column to the existing DataFrame
    """
    deepseek_client = openai.OpenAI(
        base_url="https://api.deepseek.com/v1", api_key=os.environ["DEEPSEEK_API_KEY"]
    )

    df["deepseek_translation"] = ""

    print("\nGetting DeepSeek translations...")
    for idx in tqdm(df.index):
        nahuatl_word = df.loc[idx, "nahuatl"]

        deepseek_translation = get_deepseek_translation(nahuatl_word, deepseek_client)
        df.loc[idx, "deepseek_translation"] = deepseek_translation

        time.sleep(0.5)

    return df


def main():
    input_path = "nahuatl_translations.csv"
    df = pd.read_csv(input_path)

    df_with_deepseek = add_deepseek_translations(df)

    output_path = "nahuatl_translations_with_deepseek.csv"
    df_with_deepseek.to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)
    print("\nSample of results:")
    print("=" * 80)
    print(df_with_deepseek.head())
    print(f"\nTotal pairs processed: {len(df_with_deepseek)}")


if __name__ == "__main__":
    main()
