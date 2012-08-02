package edu.umass.cs.wallach;

import java.io.*;
import java.util.*;
import java.util.zip.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.Maths;

public class WordScore {

  // observed counts

  private int[][] wordTopicCounts; // N_{w|j}
  private int[] wordTopicCountsNorm; // N_{.|j}

  private int[][] wordTopicHist;
  private int[] wordTopicNormHist;

  private int W, T; // constants

  // hyperparamters

  private double[] param;
  private double paramSum;

  // create a score function with zero counts

  public WordScore(int W, int T, double paramSum) {

    this.W = W;
    this.T = T;

    this.param = new double[W];
    Arrays.fill(this.param, paramSum / W);

    this.paramSum = paramSum;

    // allocate space for counts

    wordTopicCounts = new int[W][T];
    wordTopicCountsNorm = new int[T];
  }

  public double getScore(int w, int j) {

    int nwj = wordTopicCounts[w][j];
    int nj = wordTopicCountsNorm[j];

    return (nwj + param[w]) / (nj + paramSum);
  }

  public double getScoreNoPrior(int w, int j) {

    int nj = wordTopicCountsNorm[j];

    if (nj == 0)
      return 0.0;
    else {

      double score = (double) wordTopicCounts[w][j] / (double) nj;

      return score;
    }
  }

  public void incrementCounts(int w, int j, boolean updateHist) {

    int oldCount = wordTopicCounts[w][j]++;
    wordTopicCountsNorm[j]++;

    if (updateHist) {

      assert wordTopicHist != null;

      wordTopicHist[w][oldCount]--;
      wordTopicHist[w][oldCount + 1]++;
    }
  }

  public void decrementCounts(int w, int j, boolean updateHist) {

    int oldCount = wordTopicCounts[w][j]--;
    wordTopicCountsNorm[j]--;

    if (updateHist) {

      assert wordTopicHist != null;

      wordTopicHist[w][oldCount]--;
      wordTopicHist[w][oldCount - 1]++;
    }
  }

  public void resetCounts() {

    for (int w=0; w<W; w++)
      Arrays.fill(wordTopicCounts[w], 0);

    Arrays.fill(wordTopicCountsNorm, 0);
  }

  // computes log prob using the predictive distribution

  public double logProb(InstanceList docs, int[][] z) {

    double logProb = 0.0;

    resetCounts();

    int D = docs.size();

    assert z.length == D;

    for (int d=0; d<D; d++) {

      FeatureSequence fs = (FeatureSequence) docs.get(d).getData();

      int nd = fs.getLength();

      for (int i=0; i<nd; i++) {

        int w = fs.getIndexAtPosition(i);
        int j = z[d][i];

        logProb += Math.log(getScore(w, j));

        incrementCounts(w, j, false);
      }
    }

    return logProb;
  }

  public double[] getParam() {

    return param;
  }

  public double getParamSum() {

    return paramSum;
  }

  public void printParam(String fileName) {

    try {

      PrintWriter pw = new PrintWriter(fileName);

      for (int i=0; i<param.length; i++)
        pw.println(param[i]);

      pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  public void initializeHists(int[] wordCounts) {

    assert wordCounts.length == W;

    wordTopicHist = new int[W][];

    for (int w=0; w<W; w++) {
      wordTopicHist[w] = new int[wordCounts[w] + 1];

      for (int j=0; j<T; j++)
        wordTopicHist[w][wordTopicCounts[w][j]]++;
    }
  }

  public void optimizeParam(int numItns) {

    assert wordTopicHist != null;

    // update histograms

    int njMax = wordTopicCountsNorm[0];

    for (int j=1; j<T; j++)
      if (wordTopicCountsNorm[j] > njMax)
        njMax = wordTopicCountsNorm[j];

    wordTopicNormHist = new int[njMax + 1];

    for (int j=0; j<T; j++)
      wordTopicNormHist[wordTopicCountsNorm[j]]++;

    // optimize hyperparameters

    for (int i=0; i<numItns; i++) {

      double denominator = 0.0;
      double currentDigamma = 0.0;

      for (int n=1; n<wordTopicNormHist.length; n++) {
        currentDigamma += 1.0 / (paramSum + n - 1);
        denominator += wordTopicNormHist[n] * currentDigamma;
      }

      paramSum = 0.0;

      for (int w=0; w<W; w++) {

        double oldParam = param[w];
        param[w] = 0.0;
        currentDigamma = 0;

        for (int n=1; n<wordTopicHist[w].length; n++) {
          currentDigamma += 1.0 / (oldParam + n - 1);
          param[w] += wordTopicHist[w][n] * currentDigamma;
        }

        param[w] *= oldParam / denominator;

        if (param[w] == 0.0)
          param[w] = 1e-10;

        paramSum += param[w];
      }
    }
  }

  private double digamma(double x) {

    x = x + 6;

    double p = 1.0 / (x * x);

    p = (((0.004166666666667 * p - 0.003968253986254) * p + 0.008333333333333) * p - 0.083333333333333) * p;
    p = p + Math.log(x) - 0.5 / x - 1 / (x-1) - 1 / (x-2) - 1 / (x-3) - 1 / (x-4) - 1 / (x-5) - 1 / (x-6);

    return p;
  }

  public void optimizeParamSum(int numItns) {

    for (int i=0; i<numItns; i++) {

      double q = 0.0;
      double digammaParamSum = digamma(paramSum);

      for (int j=0; j<T; j++)
        q += digamma(wordTopicCountsNorm[j] + paramSum) - digammaParamSum;

      double p = 0.0;

      for (int w=0; w<W; w++) {

        double digammaParam = digamma(param[w]);

        for (int j=0; j<T; j++)
          p += (digamma(wordTopicCounts[w][j] + param[w]) - digammaParam) / W;
      }

      paramSum *= p / q;

      for (int w=0; w<W; w++)
        param[w] = paramSum / W;
    }
  }

  public void print(Alphabet dict, String fileName) {

    print(dict, 0.0, -1, false, fileName);
  }

  public void print(Alphabet dict, double threshold, int numWords, boolean summary, String fileName) {

    assert dict.size() == W;

    try {

      PrintStream pw = null;

      if (fileName == null)
        pw = new PrintStream(System.out, true);
      else {

        pw = new PrintStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(new File(fileName)))));

        if (!summary)
          pw.println("#topic typeindex type proportion");
      }

      Probability[] probs = new Probability[W];

      for (int j=0; j<T; j++) {
        for (int w=0; w<W; w++)
          probs[w] = new Probability(w, getScore(w, j));

        Arrays.sort(probs);

        if ((numWords > W) || (numWords < 0))
          numWords = W;

        StringBuffer line = new StringBuffer();

        for (int i=0; i<numWords; i++) {

          // break if there are no more words whose proportion is
          // greater than zero or threshold...

          if ((probs[i].prob == 0) || (probs[i].prob < threshold))
            break;

          if ((fileName == null) || summary){
            line.append(dict.lookupObject(probs[i].index));
            line.append(" ");
          }
          else {
            pw.print(j); pw.print(" ");
            pw.print(probs[i].index); pw.print(" ");
            pw.print(dict.lookupObject(probs[i].index)); pw.print(" ");
            pw.print(probs[i].prob); pw.println();
          }
        }

        String string = line.toString();

        if ((fileName == null) || summary)
          if (!string.equals(""))
            pw.println("Topic " + j + ": " + string);
      }

      if (fileName != null)
        pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }
}
