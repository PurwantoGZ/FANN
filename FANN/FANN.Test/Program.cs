using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FANN.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] trainData = new double[4][];

            #region Data Training
            trainData[0] = new double[] {1.0,1.0,1 };
            trainData[1] = new double[] {1.0,0.0,0 };
            trainData[2] = new double[] {0.0,1.0,0 };
            trainData[3] = new double[] {0.0,0.0,0 };
            #endregion

            #region Data Testing
            double[][] testku = new double[4][];
            testku[0] = new double[] { 1, 1 };
            testku[1] = new double[] { 1, 0 };
            testku[2] = new double[] { 0, 1 };
            testku[3] = new double[] { 0, 0 };
            #endregion

            const int numInput = 2;
            const int numHidden = 10;
            const int numOutput = 1;

            int maxEpoch = 1000;
            double learnrate = 0.5;
            double momentum = 0.1;

            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] weights = new double[numWeights];
            Neuron nn = new Neuron(numInput, numHidden, numOutput);

            Console.WriteLine("Creating a {0}-input, {1}-hidden, {2}-output neural network", numInput, numHidden, numOutput);
            Console.WriteLine("Inisialisasi Weights");
            nn.InitializedWeights();

            Console.WriteLine("Setting Max.Iterasi={0}, K.Belajar={1}, & Momentum={2}", maxEpoch, learnrate, momentum);
            Console.WriteLine("Training......");
            nn.TrainBP(trainData, 0.1, maxEpoch, learnrate, momentum);

            Console.WriteLine("\n------------WEIGHTS----------------");
            weights = nn.Getweights();
            nn.ShowVector(weights, 5, 3, true);
            Console.WriteLine("Testing...");
            
            nn.TestBP(testku);
           
            Console.WriteLine("Training Accuracy = {0}", nn.Accuracy(trainData).ToString("F4"));
            

            Console.ReadLine();
        }
    }
}
