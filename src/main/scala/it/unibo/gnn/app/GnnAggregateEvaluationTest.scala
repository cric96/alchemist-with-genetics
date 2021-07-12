package it.unibo.gnn.app

import io.jenetics._
import io.jenetics.engine.{Engine, EvolutionResult, EvolutionStatistics, Limits}
import io.jenetics.util.RandomRegistry
import io.jenetics.xml.Writers
import it.unibo.gnn.app.program.ScafiHopCountGNN
import it.unibo.gnn.evolutionary.{GNNCodec, JeneticsFacade}
import it.unibo.gnn.model.GraphNeuralNetwork
import it.unibo.scafi.config.GridSettings
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._

import java.io.FileOutputStream
import java.lang
import java.util.Random
import scala.jdk.CollectionConverters.IterableHasAsScala

object LinearGnnSimulationTest
  extends GnnSimulationTest(NetworkConfigurations.linearConfig) with App

object NonLinearGnnSimulationTest
  extends GnnSimulationTest(NetworkConfigurations.nonLinearConfig) with App

abstract class GnnSimulationTest(config : NetworkConfiguration) {
  // Constants
  /// Environment constants
  private val seed = 42
  private val random = new Random(seed)
  private val gridSetting = GridSettings(3, 3, 50, 50)
  /// Scafi constants
  private val radius = 60
  //// Sensors
  private val networkSensor = "network"
  private val stateSensor = "initialState"
  private val sourceSensor = "source"
  //// Sensors values
  private val initialSourceValue = 0.0f
  private val initialStateValue = (0 until config.stateSize).map(_ => -1f).toArray
  private val sourceOnValue = 1.0f
  private val sourceId = 0
  // Genetics constants
  private val steady = 50
  private val populationSize = 200
  // Utility
  private val codec = config.codec
  RandomRegistry.random(random)
  // utility for creating ScaFi simulation, return the simulator and the exports produced
  private def spawnSimulation(program: AggregateProgram, length: Int = 10, network: Option[GraphNeuralNetwork] = None): (NetworkSimulator, Map[ID, Double]) = {
    val simulator = simulatorFactory.gridLike(gridSetting, radius, seeds = Seeds(seed, seed, seed))
    network.foreach(simulator.addSensor(networkSensor, _))
    val networkSimulator = simulator.asInstanceOf[NetworkSimulator] //unsafe
    simulator.addSensor(sourceSensor, initialSourceValue)
    simulator.chgSensorValue(sourceSensor, Set(sourceId), sourceOnValue)
    simulator.addSensor(stateSensor, initialStateValue)
    val ids = (0 to length).flatMap(_ => 0 until networkSimulator.ids.size)
    ids foreach (simulator.exec(program, program.main(), _))
    val results = simulator.exports().map { case (id, data) => id -> data.get.root[Double]() }
    (networkSimulator, results)
  }

  private def statisticsByGeneration(e: EvolutionResult[DoubleGene, lang.Double]): Unit = {
    val valid = e.population().map(_.fitness()).asScala.map[Double](a => a).filter(_.isFinite)
    val meanFitness = valid.sum / valid.size
    println(s"Generation ${e.generation()}, mean fitness : ${meanFitness}, best : ${e.bestFitness()}, worst : ${valid.max}")
  }

  // fitness used to valuate the solution
  private def fitness(genotype: Genotype[DoubleGene], codec: GNNCodec): Double = {
    val gnn = codec.loadFromGenotype(genotype)
    val (_, gnnResults) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
    val fitness = gnnResults.map { case (id, v) => references(id) - v }.map { value => Math.abs(value) }.sum
    fitness
  }

  // standard hop count program
  private val hopCountProgram = new AggregateProgram with FieldUtils {
    override def main(): Any = rep(Double.PositiveInfinity) {
      value => mux(mid() == sourceId) {
        0.0
      } {
        minHoodPlus(nbr(value) + 1)
      }
    }
  }
  // create the ground truth
  private val (_, references) = spawnSimulation(hopCountProgram)
  // genetics part
  /// factory of graph neural network encoded in terms of genotype
  private val factory = codec.genotypeFactory()
  /// jenetics engine used, it describes the evolutionary algorithms that will be used.
  private val runner: Engine[DoubleGene, lang.Double] =
    JeneticsFacade.doubleEngine[DoubleGene](genotype => fitness(genotype, codec), factory)
      .populationSize(populationSize)
      .survivorsSelector(new TournamentSelector(5))
      .offspringSelector(new RouletteWheelSelector())
      .alterers(
        new Mutator(0.03),
        new MeanAlterer(0.6)
      )
      .minimizing()
      .build()
  //a mutable object used to gather evolutionary statistics
  private val statistics = EvolutionStatistics.ofNumber[lang.Double]()
  //start the evolutionary algorithm
  private val bestResult =
    RandomRegistry.`with`(random, _ => { //with is used to give the reproducibility of tests
      runner.stream()
        .limit(Limits.bySteadyFitness[lang.Double](steady))
        .peek(e => statistics.accept(e))
        .peek(statisticsByGeneration)
        .collect(EvolutionResult.toBestPhenotype[DoubleGene, lang.Double])
    })
  private val gnn = codec.loadFromGenotype(bestResult.genotype())
  private val (_, gnnResult) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
  private val file = new FileOutputStream(config.outputFile)
  Writers.Genotype.write[lang.Double, DoubleGene, DoubleChromosome](file, bestResult.genotype(), Writers.DoubleChromosome.writer())
  println(gnnResult)
  println(references)
  file.close()
}
