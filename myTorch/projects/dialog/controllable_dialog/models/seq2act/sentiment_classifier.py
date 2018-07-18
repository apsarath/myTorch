import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


class SentimentClassifier:

    def word_feats(self, words):
        return dict([(word, True) for word in words])

    def __init__(self):
        self.negative_sentiment_words = ['hate', 'dislike', "stupid", "dumb", "unintelligent", "quiet", "silence", "gibberish", "silly", "discounted", "liar", "lies", "lie", "enough", "ouch", "quit", "weird", "odd", "creepy", "wrong", "rude"]
        self.negative_sentiment_phrases = ['make any sense', 'make sense', "don 't like", "short comings", "short coming", "stop saying", "take over the world", "not very nice", "not nice", "no laugh", "your bad", "you 're bad", "you bad", "I don 't understand", "not true", "that 's not", "no sense"]

        self.positive_sentiment_words = ['good', 'great', 'amazing', 'ha', 'he', 'nice', 'welcome', 'cool']
        self.positive_sentiment_phrases = ["me too", "nicest bought", "nice bought", "nicest bot", "nice bot", "I agree", "I like"]

        self.sentiment_types = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

        # Train NLTK model on movie reviews corpus...
        nltk.download("movie_reviews")
        negids = movie_reviews.fileids('neg')
        posids = movie_reviews.fileids('pos')

        negfeats = [(self.word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
        posfeats = [(self.word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

        negcutoff = len(negfeats)
        poscutoff = len(posfeats)

        trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
        testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

        self.classifier = NaiveBayesClassifier.train(trainfeats)

    def get_movie_model_output_probabilities(self, utterance):
        output = self.classifier.prob_classify(self.word_feats(utterance))
        return output._prob_dict, output.max()

    def has_numbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def get_utterance_sentiment(self, utterance, extended_dialogue_act="OTHER"):
        if extended_dialogue_act in ["GENERIC_QUESTION", "PERSONAL_QUESTION", "POLITICS", "REQUEST", "GREETING", "PLAY_SONG", "SET_ALARM", "GOODBYE"]:
            return "NEUTRAL"

        if extended_dialogue_act == "PROFANE":
            return "NEGATIVE"

        utterance_processed = utterance.replace("'", " '").replace('.', '').replace('!', '').replace(',', '').replace('?', '')
        utterance_lower = utterance_processed.lower()
        utterance_split = utterance_lower.split()

        for word in utterance_split:
            if word in self.negative_sentiment_words:
                return "NEGATIVE"

        for negative_sentiment_phrase in self.negative_sentiment_phrases:
            if negative_sentiment_phrase in utterance_lower:
                return "NEGATIVE"

        for word in utterance_split:
            if word in self.positive_sentiment_words:
                return "POSITIVE"

        for word in utterance_split:
            if word in self.positive_sentiment_words:
                return "POSITIVE"

        for positive_sentiment_phrase in self.positive_sentiment_phrases:
            if positive_sentiment_phrase in utterance_lower:
                return "POSITIVE"

        # The NLTK classifier was trained on movie reviews,
        # where numbers have a very different meaning.
        if (self.has_numbers(utterance)) or (len(utterance_split) <= 3):
            return "NEUTRAL"

        alpha = 0.5
        min_logprob = -0.6

        sentiment_probs, _ = self.get_movie_model_output_probabilities(utterance)

        if sentiment_probs['pos'] + alpha > sentiment_probs['neg']:
            if sentiment_probs['pos'] > min_logprob:
                if not utterance_processed == "REJECT":
                    return "POSITIVE"
        elif sentiment_probs['neg'] + alpha > sentiment_probs['pos']:
            if sentiment_probs['neg'] > min_logprob:
                if not utterance_processed == "ACCEPT":
                    return "NEGATIVE"

        return "NEUTRAL"

    def get_sentiment_id(self, utterance):
        sentiment = self.get_utterance_sentiment(utterance)
        if sentiment == "NEGATIVE":
            return 0
        elif sentiment == "POSITIVE":
            return 1
        else:
            return 2

if __name__ == "__main__":
    sentiment_classifier = SentimentClassifier()

    print('sentiment_classifier', sentiment_classifier.get_utterance_sentiment("you're a dumb bot", "OTHER"))
    print('sentiment_classifier', sentiment_classifier.get_utterance_sentiment("okay okay", "OTHER"))
    print('sentiment_classifier', sentiment_classifier.get_utterance_sentiment("ha that's funny", "OTHER"))
