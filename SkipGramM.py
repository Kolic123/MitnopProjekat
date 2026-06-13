
import os

os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'


import collections
import random
import zipfile
import numpy as np
import tensorflow as tf

#%%
class TrainingConfig:
    def __init__(self):
        self.learning_rate = 0.1
        self.batch_size = 128
        self.total_iterations = 1000000
        self.log_interval = 10000
        self.eval_interval = 100000
        self.checkpoints = [ 200000, 400000, 600000, 800000, 900000]
        
        self.embedding_dimension = 200
        self.vocab_capacity = 50000 #ogranicenje zbog limita kompljutera
        self.min_word_frequency = 10
        
        self.context_radius = 3 
        self.words_per_target = 2 
        self.negative_sample_count = 128
        
        self.test_words = [b'five', b'of', b'going', b'hardware', b'american', b'britain']

config = TrainingConfig()

#%%

def load_text_sequence(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        file_content = archive.read(archive.namelist()[0]).decode('utf-8')
        return file_content.lower().split()
#%%
class WordVocabulary:
    def __init__(self, max_size, min_freq):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2id = {}
        self.id2word = {}
        self.unk_id = 0
        
    def build(self, word_list):
  
        freq_table = collections.Counter(word_list)
        frequent_words = [(w, c) for w, c in freq_table.items() if c >= self.min_freq]
        frequent_words.sort(key=lambda x: -x[1])

        top_tokens = frequent_words[:self.max_size - 1]
        
        self.word2id = {'UNK': self.unk_id}
        self.id2word = {self.unk_id: 'UNK'}
        
        for idx, (word, _) in enumerate(top_tokens, start=1):
            self.word2id[word] = idx
            self.id2word[idx] = word
    
    def encode_sequence(self, word_list):
        unknown_count = 0
        encoded = []
        for w in word_list:
            wid = self.word2id.get(w, self.unk_id)
            if wid == self.unk_id:
                unknown_count += 1
            encoded.append(wid)
        return encoded, unknown_count
    def clean_stop_words(self, word_list):
        return [word for word in word_list if len(word) >= 2 or word == 'a']
     
    
    @property
    def size(self):
        return len(self.word2id)

#%%
class SkipGramBatchGenerator:
    def __init__(self, data_sequence, window_radius, words_per_target):
        self.sequence = data_sequence
        self.radius = window_radius
        self.pairs = words_per_target
        self.window_size = 2 * window_radius + 1
        self.current_pos = 0
        self.sliding_buffer = collections.deque(maxlen=self.window_size)
        
    def _refill_buffer(self):
        if self.current_pos + self.window_size > len(self.sequence):
            self.current_pos = 0
        self.sliding_buffer.clear()
        self.sliding_buffer.extend(self.sequence[self.current_pos:self.current_pos + self.window_size])
        self.current_pos += self.window_size
    
    def _advance_window(self):
        if self.current_pos == len(self.sequence):
            self.sliding_buffer.extend(self.sequence[0:self.window_size])
            self.current_pos = self.window_size
        else:
            self.sliding_buffer.append(self.sequence[self.current_pos])
            self.current_pos += 1
    
    def generate_batch(self, batch_size):
        assert batch_size % self.pairs == 0
        
        inputs = np.zeros(batch_size, dtype=np.int32)
        outputs = np.zeros((batch_size, 1), dtype=np.int32)
        
        self._refill_buffer()
        
        center_idx = self.radius
        examples_per_batch = batch_size // self.pairs
        
        for ex in range(examples_per_batch):
            context_positions = [p for p in range(self.window_size) if p != center_idx]
            selected_context = random.sample(context_positions, self.pairs)
            
            target_word = self.sliding_buffer[center_idx]
            
            for pair_idx, ctx_pos in enumerate(selected_context):
                batch_pos = ex * self.pairs + pair_idx
                inputs[batch_pos] = target_word
                outputs[batch_pos, 0] = self.sliding_buffer[ctx_pos]
            
            self._advance_window()
        
      
        self.current_pos = (self.current_pos + len(self.sequence) - self.window_size) % len(self.sequence)
        
        return inputs, outputs

#%%
class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim, num_negatives):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        
        self.word_embeddings = tf.Variable(
            tf.random.normal([vocab_size, embedding_dim]),
            name='embeddings'
        )
        self.context_weights = tf.Variable(
            tf.random.normal([vocab_size, embedding_dim]),
            name='context_weights'
        )
        self.context_biases = tf.Variable(
            tf.zeros([vocab_size]),
            name='context_biases'
        )
    
    def get_embeddings(self, word_ids):
        return tf.nn.embedding_lookup(self.word_embeddings, word_ids)
    
    def compute_loss(self, input_embeddings, target_context):
        target_context = tf.cast(target_context, tf.int64)
        return tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.context_weights,
                biases=self.context_biases,
                labels=target_context,
                inputs=input_embeddings,
                num_sampled=self.num_negatives,
                num_classes=self.vocab_size
            )
        )
    
    def get_similarity_matrix(self, query_embeddings):
        query_norm = query_embeddings / tf.norm(query_embeddings, axis=1, keepdims=True)
        target_norm = self.word_embeddings / tf.norm(self.word_embeddings, axis=1, keepdims=True)
        return tf.matmul(query_norm, target_norm, transpose_b=True)

#%%
class SkipGramTrainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.optimizer = tf.optimizers.SGD(learning_rate)
    
    @tf.function
    def training_step(self, batch_inputs, batch_labels):
        with tf.GradientTape() as tape:
            embeddings = self.model.get_embeddings(batch_inputs)
            loss = self.model.compute_loss(embeddings, batch_labels)
        
        gradients = tape.gradient(loss, [self.model.word_embeddings, self.model.context_weights, self.model.context_biases])
        self.optimizer.apply_gradients(zip(gradients,[ self.model.word_embeddings, self.model.context_weights, self.model.context_biases]))
        return loss

#%%
def save_embeddings(model, Word_vocab, naziv):
    final_embeddings = model.word_embeddings.numpy()

    with open(naziv, 'w', encoding='utf-8') as f:
        f.write(f"{Word_vocab.size} {config.embedding_dimension}\n")
        for i in range(Word_vocab.size):
            word = Word_vocab.id2word[i]
            vec_str = ' '.join([str(x) for x in final_embeddings[i]])
            f.write(f"{word} {vec_str}\n")
#%%
def main():
    raw_text = load_text_sequence('text8_dataset/archive.zip')
    vocab = WordVocabulary(config.vocab_capacity, config.min_word_frequency)
    raw_text = vocab.clean_stop_words(raw_text)
    vocab.build(raw_text)
    encoded_text, unk_cnt = vocab.encode_sequence(raw_text)
    print(f"U vokab je: {vocab.size}, UNK ima: {unk_cnt}")
    data_gen = SkipGramBatchGenerator(
        encoded_text,
        config.context_radius,
        config.words_per_target
        )
    with tf.device('/cpu:0'):
        model = SkipGramModel(vocab.size, config.embedding_dimension, config.negative_sample_count)
        trainer = SkipGramTrainer(model, config.learning_rate)
        test_ids = np.array([vocab.word2id.get(w.decode(), 0) for w in config.test_words])
        for step in range(1, config.total_iterations + 1):
            batch_x, batch_y = data_gen.generate_batch(config.batch_size)
            loss = trainer.training_step(batch_x, batch_y)
            if step == 1 or step % config.log_interval == 0:
                print(f"Iteracija {step}: loss = {loss.numpy():.4f}")
                
                if step in config.checkpoints:
                    filename = f'skipgram_window3_neg128_without_small_words_{step//1000}k.txt'
                    save_embeddings(model, vocab, filename)
                
                if step == 1 or step % config.eval_interval == 0:
                    print("\nEvaluacija")
                    query_emb = model.get_embeddings(test_ids)
                    sim_matrix = model.get_similarity_matrix(query_emb).numpy()
                    
                    for i, test_word in enumerate(config.test_words):
                        nearest_indices = (-sim_matrix[i]).argsort()[1:9]
                        nearest_words = [vocab.id2word[idx] for idx in nearest_indices]
                        print(f"'{test_word.decode()}' → {nearest_words}")
                        print()
            if step == config.total_iterations:
                filename = 'skipgram_window3_neg128_without_small_words_end_1000k.txt'
                save_embeddings(model, vocab, filename)
#%%
if __name__ == "__main__":
    main()
    

