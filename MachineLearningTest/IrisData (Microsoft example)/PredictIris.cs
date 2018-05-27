using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningTest.IrisData
{
    public class PredictIris
    {
        public PredictIris()
        {
            var dataPath = Path.Combine("IrisData (Microsoft example)", "iris.data.txt");

            // Create a pipeline and load your data
            var pipeline = new LearningPipeline
            {
                // Aggiungo alla pipeline i dati del txt serializzati come IrisData
                new TextLoader<IrisData>(dataPath, false, ","),
                // Definisco l'output
                // Trasformo la colonna "label" che è una stringa in numeri (indici), in quanto solo numeri possono stare nel classificatore
                // Questo devo farlo per ogni ingresso o uscita che è di tipo stringa.
                new Dictionarizer("Label"),
                // Definisco gli input
                // Mette nella colonna "Features" le colonne di IrisData (usa il reflector quindi le passo come stringhe)
                new ColumnConcatenator("Features", nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth),
                    nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth)),
                // Aggiungo un algoritmo di apprendimento, voglio un classificatore in quanto voglio prevedere che tipo di fiore a partire dalle caratteristiche dei petali

                new StochasticDualCoordinateAscentClassifier(),
                // Ritrasformo la colonna "label" che lui internamente gestice come numeri (vedi il passo con il Dictionarizer) nella stringa corrispondente, in modo che in uscita dell'algoritmo io abbia i nommi dei fiori e non dei numeri random.
                // Notare ch PredictedLabel è definito come ColumnName in IrisPrediction
                new PredictedLabelColumnOriginalValueConverter {PredictedLabelColumn = "PredictedLabel"}
            };


            // una volta definito tutto effettuo il training con IrisData in ingresso e IrisPrediction in uscita.
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // Ora che ho il mio model posso utilizzarlo per effettuare delle predizioni, per esempio gli passo un IrisData e lui mi deve dire che fiore è
            var prediction = model.Predict(new IrisData
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.ReadLine();
        }
    }
}