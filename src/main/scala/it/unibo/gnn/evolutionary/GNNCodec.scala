package it.unibo.gnn.evolutionary

import io.jenetics.{DoubleChromosome, DoubleGene, Genotype}
import it.unibo.gnn.GraphNeuralNetwork
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.cpu.nativecpu.NDArray

import scala.annotation.tailrec
import scala.jdk.CollectionConverters.{IterableHasAsScala, ListHasAsScala}

case class GNNCodec(stateEvolutionShape : MultiLayerConfiguration, outputEvaluationShape : MultiLayerConfiguration, maxWeight : Int = 1) {
  private val biasWeightCount = 1
  def loadFromGenotype(genotype : Genotype[DoubleGene]) : GraphNeuralNetwork = {
    val splitPoint = stateEvolutionShape.getConfs.asScala
      .map(_.getLayer).collect { case l : DenseLayer => l}
      .map(layer => layer.getNIn + biasWeightCount)
      .sum

    val (stateChromosomes, outputChromosome) = genotype.asScala.splitAt(splitPoint.toInt )
    val stateNetwork = loadNetworkFromGenotype(JeneticsFacade.of[DoubleGene](stateChromosomes.toSeq:_*), stateEvolutionShape)
    val outputNetwork = loadNetworkFromGenotype(JeneticsFacade.of[DoubleGene](outputChromosome.toSeq:_*), outputEvaluationShape)
    GraphNeuralNetwork(stateNetwork, outputNetwork)
  }

  def genotypeFactory(): Genotype[DoubleGene] = {
    val layers : Seq[DenseLayer] = layersFromConfiguration(stateEvolutionShape) :++ layersFromConfiguration(outputEvaluationShape)
    val stateEvolutionChromosomes = layers.flatMap(layer => (0 to layer.getNIn.toInt) map { _ => DoubleChromosome.of(-maxWeight, maxWeight, layer.getNOut.toInt) } )
    JeneticsFacade.of[DoubleGene](stateEvolutionChromosomes:_*)
  }

  private def layersFromConfiguration(config : MultiLayerConfiguration) : Seq[DenseLayer] = config.getConfs.asScala
    .map(_.getLayer)
    .collect { case layer : DenseLayer => layer }
    .toSeq

  private def loadNetworkFromGenotype(genotype: Genotype[DoubleGene], config : MultiLayerConfiguration) : MultiLayerNetwork = {
    val layers = layersFromConfiguration(config)
    val slicing = layers.map(_.getNIn.toInt + biasWeightCount)
    val chromosomes = genotype.asScala.toList
    val chromosomesForLayer = partitionWith(chromosomes, slicing)

    val linearLayerWeights = chromosomesForLayer.flatMap(chromosomeForLayer => chromosomeForLayer.flatMap { chromosome => chromosome.asScala.map(_.allele())} )
    val ndFormat : Array[Float] = linearLayerWeights.toArray.map(_.floatValue())
    val network = new MultiLayerNetwork(config)
    network.init(new NDArray(ndFormat), false)
    network
  }

  private def partitionWith[E](elements : Seq[E], partitionSizes : Seq[Int]) : Seq[Seq[E]] = {
    @tailrec
    def tailRecPartition(elements : Seq[E], partitionSizes : Seq[Int], result : Seq[Seq[E]] = Seq.empty) : Seq[Seq[E]] = partitionSizes match {
      case head :: rest =>
        val (partition, restPartitions) = elements.splitAt(head)
        tailRecPartition(restPartitions, rest, Seq(partition) :++ result)
      case Nil => result
    }
    tailRecPartition(elements, partitionSizes)
  }
}
