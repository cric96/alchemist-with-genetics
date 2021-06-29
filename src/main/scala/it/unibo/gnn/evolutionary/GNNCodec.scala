package it.unibo.gnn.evolutionary

import io.jenetics.{DoubleGene, Genotype}
import it.unibo.gnn.model.GraphNeuralNetwork
import it.unibo.gnn.model.GraphNeuralNetwork.{LinearGraphNeuralNetwork, NonLinearGraphNeuralNetwork}
import org.deeplearning4j.nn.conf.MultiLayerConfiguration

trait GNNCodec extends NetworkCodec[GraphNeuralNetwork]

object GNNCodec {
  case class NonLinearGNNCodec(
                                stateEvolutionShape : MultiLayerConfiguration,
                                outputEvaluationShape : MultiLayerConfiguration,
                                maxWeight : Int = 1
                              ) extends GNNCodec {
    private val MLPCodec = MultiLayerNetworkCodec(maxWeight, stateEvolutionShape, outputEvaluationShape)

    def loadFromGenotype(genotype : Genotype[DoubleGene]) : GraphNeuralNetwork = {
      val networks = MLPCodec.loadFromGenotype(genotype)
      NonLinearGraphNeuralNetwork(networks.head, networks.tail.head)
    }

    def genotypeFactory(): Genotype[DoubleGene] = MLPCodec.genotypeFactory()
  }

  case class LinearGNNCodec(biasShape : MultiLayerConfiguration,
                            stateEvolutionShape : MultiLayerConfiguration,
                            outputEvaluationShape : MultiLayerConfiguration,
                            maxWeight : Int = 1,
                            mu : Double = 1) extends GNNCodec {
    private val MLPCodec = MultiLayerNetworkCodec(maxWeight, biasShape, stateEvolutionShape, outputEvaluationShape)

    def loadFromGenotype(genotype : Genotype[DoubleGene]) : GraphNeuralNetwork = {
      val networks = MLPCodec.loadFromGenotype(genotype)
      LinearGraphNeuralNetwork(networks.head, networks.tail.head, networks.tail.tail.head, mu)
    }

    def genotypeFactory(): Genotype[DoubleGene] = MLPCodec.genotypeFactory()
  }

}

