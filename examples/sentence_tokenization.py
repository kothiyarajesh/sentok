import sentok

# Display current weights used by the tokenizer
# Uncomment the following line to view the current weights in use:
# print(sentok.get_weights())

# Adjust weights only if necessary for specific use cases
# For example, updating the set of start characters:
# sentok.set_weights({'start_chars': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')})

# Sample text for sentence tokenization
text = """Natural language processing (NLP) is a captivating domain that merges computer science, artificial intelligence, and linguistics. It empowers computers to comprehend, interpret, and produce human language in a manner that is both useful and insightful. NLP finds application in various fields, such as text analysis, speech recognition, and machine translation. For example, advanced language models like GPT-3 have showcased exceptional skills in generating text that resembles human writing and in answering queries. As technology progresses, NLP continues to advance, enhancing its precision and expanding its scope of applications."""

# Tokenize the sample text into sentences using the default threshold of 0.65
# You can adjust the threshold based on the quality and characteristics of your text.
sentences = sentok.sent_tokenize(text, 0.64)

# Print each extracted sentence
for sentence in sentences:
    print('->', sentence)

# Print the total number of sentences extracted
print('Total Sentences:', len(sentences))

# To obtain a DataFrame with tokenization features for further analysis or model training:
df = sentok.get_sent_tokenize_df(text)
print(df)
