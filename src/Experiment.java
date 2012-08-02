package edu.umass.cs.wallach;

import java.io.*;
import java.util.*;

import gnu.trove.*;

import cc.mallet.types.*;

public class Experiment {

  public static void main(String[] args) throws java.io.IOException {

    if (args.length != 8) {
      System.out.println("Usage: Experiment <instance_list> <num_topics> <num_itns> <print_interval> <save_state_interval> <symmetric> <optimize> <output_dir>");
      System.exit(1);
    }

    int index = 0;

    String instanceListFileName = args[index++];

    int T = Integer.parseInt(args[index++]); // # of topics

    int numIterations = Integer.parseInt(args[index++]); // # Gibbs iterations
    int printInterval = Integer.parseInt(args[index++]); // # iterations between printing out topics
    int saveStateInterval = Integer.parseInt(args[index++]);

    assert args[index].length() == 2;
    boolean[] symmetric = new boolean[2];

    for (int i=0; i<2; i++)
      switch(args[index].charAt(i)) {
      case '0': symmetric[i] = false; break;
      case '1': symmetric[i] = true; break;
      default: System.exit(1);
      }

    index++;

    assert args[index].length() == 2;
    boolean[] optimize = new boolean[2]; // whether to optimize hyperparameters

    for (int i=0; i<2; i++)
      switch(args[index].charAt(i)) {
      case '0': optimize[i] = false; break;
      case '1': optimize[i] = true; break;
      default: System.exit(1);
      }

    index++;

    String outputDir = args[index++]; // output directory

    assert index == 8;

    // load data

    InstanceList docs = InstanceList.load(new File(instanceListFileName));

    Alphabet wordDict = docs.getDataAlphabet();

    int W = wordDict.size();

    double alpha = 0.1 * T;
    double beta = 0.01 * W;

    // form output filenames

    String optionsFileName = outputDir + "/options.txt";

    String documentTopicsFileName = outputDir + "/doc_topics.txt.gz";
    String topicWordsFileName = outputDir + "/topic_words.txt.gz";
    String topicSummaryFileName = outputDir + "/topic_summary.txt.gz";
    String stateFileName = outputDir + "/state.txt.gz";
    String alphaFileName = outputDir + "/alpha.txt";
    String betaFileName = outputDir + "/beta.txt";
    String logProbFileName = outputDir + "/log_prob.txt";

    PrintWriter pw = new PrintWriter(optionsFileName);

    pw.println("Instance list = " + instanceListFileName);

    int corpusLength = 0;

    for (int d=0; d<docs.size(); d++) {

      FeatureSequence fs = (FeatureSequence) docs.get(d).getData();
      corpusLength += fs.getLength();
    }

    pw.println("# tokens = " + corpusLength);

    pw.println("T = " + T);
    pw.println("# iterations = " + numIterations);
    pw.println("Print interval = " + printInterval);
    pw.println("Save state interval = " + saveStateInterval);
    pw.println("Symmetric alpha = " + symmetric[0]);
    pw.println("Symmetric beta = " + symmetric[1]);
    pw.println("Optimize alpha = " + optimize[0]);
    pw.println("Optimize beta = " + optimize[1]);
    pw.println("Date = " + (new Date()));

    pw.close();

    LDA lda = new LDA();

    lda.estimate(docs, null, 0, T, alpha, beta, numIterations, printInterval, saveStateInterval, symmetric, optimize, documentTopicsFileName, topicWordsFileName, topicSummaryFileName, stateFileName, alphaFileName, betaFileName, logProbFileName);

  }
}
