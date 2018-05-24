using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningTest.Segmenti
{
    public class PredictDigit
    {
        public PredictDigit()
        {
            var pipeline = new LearningPipeline();
            var dataPath = Path.Combine("Segmenti", "segments.txt");
            pipeline.Add(new TextLoader<Digit>(dataPath, false, ","));
            pipeline.Add(new ColumnConcatenator("Features", nameof(Digit.Up), nameof(Digit.Middle),
                nameof(Digit.Bottom), nameof(Digit.UpLeft), nameof(Digit.BottomLeft), nameof(Digit.TopRight),
                nameof(Digit.BottomRight)));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            var model = pipeline.Train<Digit, DigitPrediction>();
            var prediction = model.Predict(new Digit
            {
                Up = 1,
                Middle = 1,
                Bottom = 1,
                UpLeft = 0,
                BottomLeft = 0,
                TopRight = 1,
                BottomRight = 0
            });

            Console.WriteLine($"Predicted digit is: {prediction.Score}, {prediction.ExpectedDigit - 1}");
            Console.ReadLine();
        }
    }
}