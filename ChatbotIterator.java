package chatbot;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class ChatbotIterator implements DataSetIterator {
    private final WordVectors wordVectors;
    private final int batch;
    private final int vectorSize;
    private final int numOuputs;
    private final List<String> labels;
    private final int truncateLength;

    private int cursor = 0;
    private final File questionsFile;
    private List<String> questions;
    private List<Integer> intents; // contains the index of the 1 hot output class per training example
    private final TokenizerFactory tokenizerFactory;

    public ChatbotIterator(WordVectors wordVectors,
                           int batch,
                           List<String> labels,
                           int truncateLength,
                           File file) {
        this.batch = batch;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.numOuputs = labels.size();
        this.labels = labels;
        this.questionsFile = file;

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        try {
            fileToList(questionsFile);
        } catch(Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void fileToList(File file) throws Exception {
        Scanner scanner = new Scanner(file);
        scanner.useDelimiter("\n");

        List<String> quests = new ArrayList<>();
        List<Integer> intents_parsed = new ArrayList<>();
        while(scanner.hasNext()) {
            String sentence = scanner.next();
            String[] parts = sentence.split("\t");
            intents_parsed.add(Integer.parseInt(parts[0]));
            quests.add(parts[1]);
        }
        this.questions = quests;
        this.intents = intents_parsed;
    }

    @Override
    public DataSet next(int num) {
        if(cursor >= questions.size()) throw new NoSuchElementException();
        try {
            return nextDataSet(num);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {
        // load questions into List of strings
        List<String> sampleQuestions = new ArrayList<>(num);
        List<Integer> sampleIntents = new ArrayList<>(num);
        for(int i = 0; i < num && cursor < totalExamples(); i++) {
            sampleQuestions.add(questions.get(cursor));
            sampleIntents.add(intents.get(cursor));
            cursor++;
        }

        // tokenize reviews and remove unknown words
        List<List<String>> allTokens = new ArrayList<>(sampleQuestions.size());
        int maxLength = 0;
        for(String question : sampleQuestions) {
            List<String> tokens = tokenizerFactory.create(question).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens) {
                if(wordVectors.hasWord(t)) {
                    tokensFiltered.add(t);
                }
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength, tokensFiltered.size());
         }

        if(maxLength > truncateLength) maxLength = truncateLength;

        INDArray features = Nd4j.create(new int[]{sampleQuestions.size(), vectorSize, maxLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{sampleQuestions.size(), numOuputs, maxLength}, 'f');

        INDArray featuresMask = Nd4j.zeros(sampleQuestions.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(sampleQuestions.size(), maxLength);

        int[] temp = new int[2];
        for(int i = 0; i < sampleQuestions.size(); i++) {
            List<String> tokens = allTokens.get(i);
            temp[0] = i;

            // get word vectors for each word in question
            // i is the sentence number and j is the word number in the sentence
            for(int j = 0; j < tokens.size() && j < maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                //
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);
            }

            int idx = sampleIntents.get(i); // the index of the 1 hot class
            int lastIdx = Math.min(tokens.size(), maxLength);

            labels.putScalar(new int[]{i, idx, lastIdx-1}, 1.0);
            labelsMask.putScalar(new int[]{i, lastIdx-1}, 1.0);
        }

        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    @Override
    public int totalExamples() {
        return (int)this.questions.size();
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return numOuputs;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batch;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batch);
    }

    public INDArray loadFeaturesFromString(String question, int maxLength) {
        List<String> tokens = tokenizerFactory.create(question).getTokens();
        List<String> tokensFiltered = new ArrayList<>();
        for(String t : tokens) {
            if(wordVectors.hasWord(t)) {
                tokensFiltered.add(t);
            }
        }

        int outputLength = Math.max(maxLength, tokensFiltered.size());
        INDArray features = Nd4j.create(1, vectorSize, outputLength);

        for(int j = 0; j < tokens.size() && j < maxLength; j++) {
            String token = tokens.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            features.put(new INDArrayIndex[]{
                    NDArrayIndex.point(0),
                    NDArrayIndex.all(),
                    NDArrayIndex.point(j)
            }, vector);
        }

        return features;
    }
}
