# FANN
Fast Artificial Neural Network Using C# Laguage
Created by PurwantoGZ

#Arsitektur Backpropagation
# 
![alt tag](https://github.com/PurwantoGZ/FANN/blob/master/arcBP.png)
#
# How to USE
1. Inisialisasikan Jumlah Sel Input,Hidden, dan Output
    const int numInput = 2
    const int numHidden = 10
    const int numOutput = 1

2. Inisialisasikan Data Training dan Data Testing
    double[][] trainData = new double[4][]
    DATA TRAINING/PENGUJAIN
    trainData[0] = new double[] {1.0,1.0,1 }// 1.0 dan 1.0 adalah input , 1 adalah target
    trainData[1] = new double[] {1.0,0.0,0 }
    trainData[2] = new double[] {0.0,1.0,0 }
    trainData[3] = new double[] {0.0,0.0,0 }

    *) NB
      * Jumlah Input data disesuaikan dengan Num Input dan Jumlah Output data disesuaikan dengan Num Output
      * Jumlah Hidden layer dinamis, Tidak ada batasan untuk jumlah Hidden layer
    
    DATA TESTING/PENGUJIAN  
    double[][] testku = new double[4][]
    testku[0] = new double[] { 1, 1 }
    testku[1] = new double[] { 1, 0 }
    testku[2] = new double[] { 0, 1 }
    testku[3] = new double[] { 0, 0 }
    
3. Tentukan Max Iterasi, Konstanta Belajar, dan Besarnya Momentum
    int maxEpoch = 1000
    double learnrate = 0.5
    double momentum = 0.1
    
    *) Disini Saya menggunakan Momentum
    *) Momentum digunakan untuk meningkatkan Proses Belajar

4. Buat Object Class dari Clas Fann
   FANN nn = new FANN(numInput, numHidden, numOutput)
   Console.WriteLine("Creating a {0}-input, {1}-hidden, {2}-output neural network", numInput, numHidden, numOutput)
   Console.WriteLine("Inisialisasi Weights")
   nn.InitializedWeights()

    Console.WriteLine("Setting Max.Iterasi={0}, K.Belajar={1}, & Momentum={2}", maxEpoch, learnrate, momentum)
    Console.WriteLine("Training......")
    nn.TrainBP(trainData, 0.1, maxEpoch, learnrate, momentum)

    Console.WriteLine("\n------------WEIGHTS----------------")
    weights = nn.Getweights()
    nn.ShowVector(weights, 5, 3, true)
    Console.WriteLine("Testing...")
            
    nn.TestBP(testku)
           
    Console.WriteLine("Training Accuracy = {0}", nn.Accuracy(trainData).ToString("F4"))
