import re
import pandas as pd 
import numpy as np

def concat_data ():
    df1 = pd.read_excel("Arabic_names.xlsx")
    df2 = pd.read_excel("all_arabic_names_preprocessed.xlsx")
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_excel("all_arabic_names_concat.xlsx", index=False)


def unify_gender(df):
    df['Gender'] = df['Gender'].str.strip().str.lower()  # clean spaces, lowercase
    df['Gender'] = df['Gender'].replace({
        'f': 'female',
        'm': 'male',
        'female': 'female',
        'male': 'male',
        'ذكر': 'male',
        "انثى": 'female'
    })
    return df


def normalize_name(name):
    """Normalize Arabic names by removing diacritics, Tatweel, and unifying letters"""
    name = name.split()[0]
    name = str(name).strip()

    # Remove diacritics
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    name = re.sub(diacritics_pattern, '', name)
    # Remove Tatweel (ــ)
    name = re.sub(r'ـ', '', name)

    # Unify Arabic letters
    name = name.replace("ى", "ي")
    name = name.replace("ة", "ه")
    name = name.replace("ئ", "ي")
    name = name.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    # Remove numbers and special characters (keep Arabic and English letters and spaces)
    name = re.sub(r'[^ء-ي\s]', '', name)

    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)

    return name

def removing_empty_gender(df):
    df = df[df['Gender'].notna()]                  
    df = df[df['Gender'] != '']                    
    df = df[df['Gender'] != 'nan']
    return df 


if __name__ == "__main__":
    concat_data()
    df = pd.read_excel(r"all_arabic_names_concat.xlsx")
    df = unify_gender(df)
    df = removing_empty_gender(df)
    df = df.dropna(subset=['Name', 'Gender'])
    df = df.drop_duplicates()
    df.to_excel("all_arabic_names_preprocessed2.xlsx", index=False)
    df['Name'] = df['Name'].apply(normalize_name)
    print("Data preprocessing completed.")
   