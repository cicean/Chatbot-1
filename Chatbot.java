package chatbot;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.*;

public class Chatbot {
    public static void main(String[] args) throws Exception {
        // train the rnn
        int vectorSize = 100; // the size of the word vector
        int nEpochs = 1000;
        int outputClasses = 9;

        System.out.println("Configuring the network");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER_FAN_IN)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.005)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(40)
                    .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nIn(40).nOut(30)
                     .activation(Activation.TANH).build())
                .layer(2, new GravesLSTM.Builder().nIn(30).nOut(20)
                     .activation(Activation.TANH).build())
                .layer(3, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(20).nOut(outputClasses).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        System.out.println("Loading the word vector mappings");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File("/home/zk/words-nlp/word-vectors.txt"));

        System.out.println("Mapping training data");
        List<String> labels = new ArrayList<>(Arrays.asList(
            "Hello",                            // 0
            "How are you",                      // 1
            "Where did you go to school",       // 2
            "Why programming",                  // 3
            "what projects",                    // 4
            "why should we hire you",           // 5
            "What are your strengths",          // 6
            "what are your weaknesses",         // 7
            "what is your major"                // 8
        ));
        ChatbotIterator train = new ChatbotIterator(wordVectors, 32, labels, 15, new File("/home/zk/words-nlp/chatbot-questions.txt"));

        System.out.println("Training the network");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(train);
            train.reset();
            System.out.print((i+1) + ": ");
            // test for network percentage and break if accuracy high enough
            if(testNetwork(train, net) > 0.95) break;
        }
    }

    public static float testNetwork(ChatbotIterator train, MultiLayerNetwork net) {
        int correct = 0;
        List<String> sentences = new ArrayList<>(Arrays.asList(
            "Hello",
            "Where did you attend school?",
            "What about programming do you like?",
            "How are you?",
            "Tell me why I should hire you.",
            "Talk about your weaknesses.",
            "What are your strengths?",
            "What projects have you been working on?",
            "Hi",
            "What school did you go to?",
            "What is your biggest weakness?",
            "How have you been recently",
            "Why should we hire you instead of someone else?",
            "What have you done to improve yourself?",
            "Nice to meet you.",
            "What school did you graduate from?",
            "What was your major",
            "What school be you attending?",
            "What did you major in?",
            "What college did you go to?"
        ));
        List<Integer> labels = new ArrayList<>(Arrays.asList(
           0, 2, 3, 1, 5, 7, 6, 4, 0, 2, 7, 1, 5, 4, 0, 2, 8, 2, 8, 2
        ));
        int total = labels.size();
        for(int i = 0; i < sentences.size(); i++) {
            if(testSentence(sentences.get(i), train, net, labels.get(i))){
                correct++;
            }
        }
        System.out.println(correct + "/" + total);
        return (float)correct/(float)total;
    }

    public static boolean testSentence(String sentence, ChatbotIterator train, MultiLayerNetwork net, int label) {
        INDArray testMatrix = train.loadFeaturesFromString(sentence, 10);
        INDArray output = net.output(testMatrix);
        int timeSeriesLength = output.size(2);
        // int timeSeriesLength = sentence.split(" ").length;
        INDArray probabilitiesAtLastWord = output.get(
                NDArrayIndex.point(0),
                NDArrayIndex.all(),
                NDArrayIndex.point(timeSeriesLength - 1));
        int mIndex = 0;
        float max = 0;
        for(int i = 0; i < probabilitiesAtLastWord.length(); i++) {
            if(probabilitiesAtLastWord.getFloat(i) > max) {
                max = probabilitiesAtLastWord.getFloat(i);
                mIndex = i;
            }
        }
        if(label == mIndex) {
            return true;
        } else {
            return false;
        }
    }
}