import numpy as np


class NotInVocabularyError(Exception):
    """Auxillary class."""
    pass


class GloVeDataIterator():
    """
    Args:
        get_text_iterator: Callable that returns an iterator that yields
            text, as a list of words (strings).
        window_size: Positive integer.
        min_word_count: Non-negative integer.
    """

    def __init__(self,
                 get_text_iterator,
                 window_size,
                 min_word_count=6):
        self.get_text_iterator = get_text_iterator
        self.window_size = window_size
        self.min_word_count = min_word_count

        # Get vocabulary first, and then the co-occurance counts
        self._vocab = self._get_vocabulary()
        self._X = self._get_cooccurance_counts()

        self.vocab_size = len(self._vocab)

    def _get_vocabulary(self):
        """Returns a dict mapping from word (string) to its ID (as integer)
        in vocabulary."""
        word_counts = {}
        for text in self.get_text_iterator():
            for word in text:
                try:
                    word_counts[word] += 1
                except KeyError:
                    word_counts[word] = 1
        word_counts = [(w, c) for w, c in word_counts.items()
                       if c >= self.min_word_count]
        word_counts = sorted(word_counts, key=lambda _: _[1], reverse=True)
        vocab = {w: i for i, (w, c) in enumerate(word_counts)}
        return vocab

    def _get_cooccurance_counts(self):
        r"""Returns a dict mapping from word-word ID pair :math:`(i, j)` to
        its co-occurance counts :math:`X_ij`."""
        # Stats the word-word co-occurance counts
        X = {}  # initialize.

        def count_in_text(text):
            window = []  # initialize.
            for word in text:
                window.append(word)
                if len(window) > 2 * self.window_size + 1:
                    window = window[1:]
                for other_word in window[:-1]:
                    try:
                        ij = (self.get_word_id(word),
                              self.get_word_id(other_word))
                        try:
                            X[ij] += 1
                        except KeyError:
                            X[ij] = 1
                    except NotInVocabularyError:
                        pass

        for text in self.get_text_iterator():
            count_in_text(text)

        return X

    def get_word_id(self, word):
        """Returns the ID of the `word` in the vocabulary.

        Args:
            word: String.

        Returns:
            Non-negative integer in the range [0, self.vocab_size).

        Raise:
            NotInVocabularyError.
        """
        try:
            return self._vocab[word]
        except KeyError:
            raise NotInVocabularyError

    def X(self, i, j):
        """Returns the word-word co-occurance counts of the i-th and the j-th
        words in the vocabulary.

        Args:
            i: Non-negative integer in the range [0, self.vocab_size).
            j: Non-negative integer in the range [0, self.vocab_size).

        Returns:
            Non-negative integer.
        """
        assert i < self.vocab_size and j < self.vocab_size
        try:
            return self._X[(i, j)]
        except KeyError:
            return 0

    def __iter__(self):
        """
        Yields:
          Numpy array with shape `[None, 3]` and int-dtype, where the first
          column is for `i`, the second for `j` and the final one for `X_ij`.
        """
        while True:
            i, j = np.random.randint(0, self.vocab_size - 1, size=[2])
            yield np.array([i, j, self.X(i, j)])


if __name__ == '__main__':

    """Test"""

    import os
    from hellonlp.glove.corpus import get_wiki_corpus

    # Get file path
    script_path = os.path.abspath(__file__)
    data_dir = os.path.join(script_path, '../../../dat')
    file_name = 'enwiki-latest-pages-articles1.xml-p10p30302.bz2'
    file_path = os.path.join(data_dir, file_name)

    wiki_corpus = get_wiki_corpus(file_path)

    def get_text_iterator(corpus=wiki_corpus):
        return corpus.get_texts()

    glove_data_iter = GloVeDataIterator(get_text_iterator, window_size=2)

    # Display results
    print('{} words in vocabulary.'.format(glove_data_iter.vocab_size))
    non_vanishing_counts = len(
        [v for k, v in glove_data_iter._X.items() if v > 0])
    print('{} non-vanishing counts in the word-word co-occurance counts.'
          .format(non_vanishing_counts))
