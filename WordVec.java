package chatbot;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.util.Collection;

public class WordVec {
    public static void main(String[] args) throws Exception {
        // setup word2vec
        // the iterator seperates sentences using a newline character
        SentenceIterator iter = new LineSentenceIterator(new File("/home/zk/words-nlp/sentences.txt"));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        // split on white spaces
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        // build the model
        System.out.println("Building the model");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)  // exclude words we can't build an accurate context for
                .iterations(2)
                .epochs(10)
                .layerSize(100)  // the number of features in the word vector
                .seed(42)
                .windowSize(5)   // rolling skip gram window size
                .iterate(iter)   // the input sentences
                .tokenizerFactory(t)  // the tokenizer
                .build();

        // fit the model (train)
        System.out.println("Training the model");
        vec.fit();

        System.out.println("Writing the model to file");
        // write all word vectors to a file, this model can be reloaded from here
        WordVectorSerializer.writeWordVectors(vec, "/home/zk/words-nlp/word-vectors.txt");

        Collection<String> lst = vec.wordsNearest("college", 5);
        System.out.println(lst);

        // lookup vector for word
        double[] wordVector = vec.getWordVector("job");
    }
}
