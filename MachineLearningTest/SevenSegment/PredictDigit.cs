using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningTest.SevenSegment
{
    public class PredictDigit
    {
        public PredictDigit()
        {
            var pipeline = new LearningPipeline();
            var dataPath = Path.Combine("Segmenti", "segments.txt");
            pipeline.Add(new TextLoader<Digit>(dataPath, false, ","));
            pipeline.Add(new ColumnConcatenator("Features", nameof(Digit.Features)));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            var model = pipeline.Train<Digit, DigitPrediction>();
            var prediction = model.Predict(new Digit
            {
                Up = 1,
                Middle = 1,
                Bottom = 0,
                UpLeft = 1,
                BottomLeft = 1,
                TopRight = 1,
                BottomRight = 1
            });

            Console.WriteLine($"Predicted digit is: {prediction.ExpectedDigit - 1}");
            Console.ReadLine();
        }
    }
}