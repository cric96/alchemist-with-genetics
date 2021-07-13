package it.unibo.gnn.evolutionary

import io.jenetics.{DoubleChromosome, DoubleGene, Genotype}
import it.unibo.gnn.model.NDArrayUtils.ndarray
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.cpu.nativecpu.NDArray

import scala.annotation.tailrec
import scala.jdk.CollectionConverters.{CollectionHasAsScala, IterableHasAsScala}

case class MultiLayerNetworkCodec(maxWeight : Double, networks : MultiLayerConfiguration *) extends NetworkCodec[Seq[MultiLayerNetwork]] {
  private val biasWeightCount = 1
  def loadFromGenotype(genotype : Genotype[DoubleGene]) : Seq[MultiLayerNetwork] = {
    val splitPoints = networks.map(_.getConfs.asScala)
      .map(layers => layers.map(_.getLayer).collect { case l : DenseLayer => l })
      .map(layers => layers.map(_.getNIn.toInt + biasWeightCount))
      .map(layers => layers.sum)

    val networkRepresentation = partitionWith(genotype.asScala.toSeq, splitPoints.toList)
    networkRepresentation.zip(networks).map { case (rep, conf) => loadNetworkFromGenotype(JeneticsFacade.of[DoubleGene](rep:_*), conf)}
  }

  def genotypeFactory(): Genotype[DoubleGene] = {
    val layers : Seq[DenseLayer] = networks.flatMap(layersFromConfiguration)
    val chromosomes = layers.flatMap(layer => (0 to layer.getNIn.toInt) map { _ => DoubleChromosome.of(-maxWeight, maxWeight, layer.getNOut.toInt) } )
    JeneticsFacade.of[DoubleGene](chromosomes:_*)
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
    network.init(ndarray(ndFormat), false)
    network
  }

  private def partitionWith[E](elements : Seq[E], partitionSizes : Seq[Int]) : Seq[Seq[E]] = {
    @tailrec
    def tailRecPartition(elements : Seq[E], partitionSizes : Seq[Int], result : Seq[Seq[E]] = Seq.empty) : Seq[Seq[E]] = partitionSizes match {
      case head :: rest =>
        val (partition, restPartitions) = elements.splitAt(head)
        tailRecPartition(restPartitions, rest, result ++ Seq(partition))
      case Nil => result
    }
    tailRecPartition(elements, partitionSizes)
  }
}

object MultiLayerNetworkCodec {
  def bounded(networks : MultiLayerConfiguration *) : MultiLayerNetworkCodec = MultiLayerNetworkCodec(1, networks:_*)
}