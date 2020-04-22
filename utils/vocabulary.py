class Vocabulary:
  def __init__(self, vocab, unk_id, sos_id, eos_id, pad_id):
    self._vocab = vocab
    self._unk_id = unk_id
    self._sos_id = sos_id
    self._eos_id = eos_id
    self._pad_id = pad_id
    
  def __len__(self):
    return len(self._vocab) + 4

  def word_to_id(self, word):
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id
    
  def sos(self):
    return self._sos_id
    
  def eos(self):
    return self._eos_id

  def pad(self):
    return self._pad_id