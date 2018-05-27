using System;
using MachineLearningTest.IrisData;
using MachineLearningTest.SevenSegment;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningTest
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //new PredictIris();
            new PredictDigit();
        }
    }
}