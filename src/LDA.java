package edu.umass.cs.wallach;

import java.io.*;
import java.util.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.*;

public class LDA {

  // observed counts

  private WordScore wordScore;
  private TopicScore topicScore;

  private int W, T, D; // constants

  private int[][] z; // topic assignments

  private LogRandoms rng; // random number generator

  private double getScore(int w, int j, int d) {

    return wordScore.getScore(w, j) * topicScore.getScore(j, d);
  }

  // computes P(w, z) using the predictive distribution

  private double logProb(InstanceList docs) {

    double logProb = 0;

    wordScore.resetCounts();
    topicScore.resetCounts();

    for (int d=0; d<D; d++) {

      FeatureSequence fs = (FeatureSequence) docs.get(d).getData();

      int nd = fs.getLength();

      for (int i=0; i<nd; i++) {

        int w = fs.getIndexAtPosition(i);
        int j = z[d][i];

        logProb += Math.log(getScore(w, j, d));

        wordScore.incrementCounts(w, j, false);
        topicScore.incrementCounts(j, d);
      }
    }

    return logProb;
  }

  private void sampleTopics(InstanceList docs, boolean init) {

    // resample topics

    int ndMax = -1;

    int[] wordCounts = (init) ? new int[W] : null;

    for (int d=0; d<D; d++) {

      FeatureSequence fs = (FeatureSequence) docs.get(d).getData();

      int nd = fs.getLength();

      if (init) {

        z[d] = new int[nd];

        if (nd > ndMax)
          ndMax = nd;
      }

      for (int i=0; i<nd; i++) {

        int w = fs.getIndexAtPosition(i);
        int oldTopic = z[d][i];

        if (!init) {
          wordScore.decrementCounts(w, oldTopic, !init);
          topicScore.decrementCounts(oldTopic, d);
        }

        // build a distribution over topics

        double dist[] = new double[T];
        double distSum = 0.0;

        for (int j=0; j<T; j++) {

          double score = getScore(w, j, d);

          dist[j] = score;
          distSum += score;
        }

        int newTopic = rng.nextDiscrete(dist, distSum);

        z[d][i] = newTopic;

        wordScore.incrementCounts(w, newTopic, !init);
        topicScore.incrementCounts(newTopic, d);

        if (init)
          wordCounts[w]++;
      }
    }

    if (init) {
      wordScore.initializeHists(wordCounts);
      topicScore.initializeHists(ndMax);
    }
  }

  public void printState(InstanceList docs, int[][] z, String file) {

    try {

      PrintWriter pw = new PrintWriter(file, "UTF-8");

      pw.println("#doc pos typeindex type topic");

      for (int d=0; d<D; d++) {

        FeatureSequence fs = (FeatureSequence) docs.get(d).getData();

        int nd = fs.getLength();

        for (int i=0; i<nd; i++) {

          int w = fs.getIndexAtPosition(i);

          pw.print(d); pw.print(" ");
          pw.print(i); pw.print(" ");
          pw.print(w); pw.print(" ");
          pw.print(docs.getDataAlphabet().lookupObject(w)); pw.print(" ");
          pw.print(z[d][i]); pw.println();
        }
      }

      pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  // estimate topics

  public void estimate(InstanceList docs, int[][] zInit, int itnOffset, int T, double alpha, double beta, int numItns, int printInterval, int saveStateInterval, boolean[] symmetric, boolean[] optimize, String documentTopicsFileName, String topicWordsFileName, String topicSummaryFileName, String stateFileName, String alphaFileName, String betaFileName, String logProbFileName) {

    boolean append = false;

    if (zInit == null)
      assert itnOffset == 0;
    else {
      assert itnOffset >= 0;
      append = true;
    }

    Alphabet wordDict = docs.getDataAlphabet();

    assert (saveStateInterval == 0) || (numItns % saveStateInterval == 0);

    rng = new LogRandoms();

    this.T = T;

    W = wordDict.size();
    D = docs.size();

    System.out.println("Num docs: " + D);
    System.out.println("Num words in vocab: " + W);
    System.out.println("Num topics: " + T);

    wordScore = new WordScore(W, T, beta);
    topicScore = new TopicScore(T, D, alpha);

    if (zInit == null) {

      z = new int[D][];
      sampleTopics(docs, true); // initialize topic assignments
    }
    else {

      z = zInit;

      int ndMax = -1;

      int[] wordCounts = new int[W];

      for (int d=0; d<D; d++) {

        FeatureSequence fs = (FeatureSequence) docs.get(d).getData();

        int nd = fs.getLength();

        if (nd > ndMax)
          ndMax = nd;

        for (int i=0; i<nd; i++) {

          int w = fs.getIndexAtPosition(i);
          int topic = z[d][i];

          wordScore.incrementCounts(w, topic, false);
          topicScore.incrementCounts(topic, d);

          wordCounts[w]++;
        }

        wordScore.initializeHists(wordCounts);
        topicScore.initializeHists(ndMax);
      }
    }

    long start = System.currentTimeMillis();

    try {

      PrintWriter logProbWriter = new PrintWriter(new FileWriter(logProbFileName, append), true);

      // count matrices have been populated, every token has been
      // assigned to a single topic, so Gibbs sampling can start

      for (int s=1; s<=numItns; s++) {

        if (s % 10 == 0)
          System.out.print(s);
        else
          System.out.print(".");

        System.out.flush();

        sampleTopics(docs, false);

        if (optimize[0]) {
          if (symmetric[0])
            topicScore.optimizeParamSum(5);
          else
            topicScore.optimizeParam(5);
        }

        if (optimize[1]) {
          if (symmetric[1])
            wordScore.optimizeParamSum(5);
          else
            wordScore.optimizeParam(5);
        }

        if (printInterval != 0) {
          if (s % printInterval == 0) {
            System.out.println();
            wordScore.print(wordDict, 0.0, 10, true, null);

            logProbWriter.println(logProb(docs));
            logProbWriter.flush();
          }
        }

        if ((saveStateInterval != 0) && (s % saveStateInterval == 0)) {
          if (stateFileName != null)
            printState(docs, z, stateFileName + "." + (itnOffset + s));
          if (alphaFileName != null)
            topicScore.printParam(alphaFileName + "." + (itnOffset + s));
          if (betaFileName != null)
            wordScore.printParam(betaFileName + "." + (itnOffset + s));
        }
      }

      Timer.printTimingInfo(start, System.currentTimeMillis());

      if (saveStateInterval == 0) {
        if (stateFileName != null)
          printState(docs, z, stateFileName);
        if (alphaFileName != null)
          topicScore.printParam(alphaFileName);
        if (betaFileName != null)
          wordScore.printParam(betaFileName);
      }

      if (documentTopicsFileName != null)
        topicScore.print(docs, documentTopicsFileName);
      if (topicWordsFileName != null)
        wordScore.print(wordDict, topicWordsFileName);

      if (topicSummaryFileName != null)
        wordScore.print(wordDict, 0.0, 10, true, topicSummaryFileName);

      logProbWriter.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  public double[] getAlpha() {

    return topicScore.getParam();
  }

  public double[] getBeta() {

    return wordScore.getParam();
  }

  public int[][] getTopics() {

    return z;
  }
}
