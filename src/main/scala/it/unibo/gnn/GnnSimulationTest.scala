package it.unibo.gnn
import io.jenetics._
import io.jenetics.engine.{Engine, EvolutionResult, EvolutionStatistics, Limits}
import io.jenetics.util.RandomRegistry
import io.jenetics.xml.Writers
import it.unibo.gnn.evolutionary.{GNNCodec, JeneticsFacade}
import it.unibo.scafi.config.GridSettings
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import java.io.FileOutputStream
import java.lang
import java.util.Random
import scala.jdk.CollectionConverters.IterableHasAsScala
import NetworkConfiguration._
object GnnSimulationTest extends App {
  val seed = 42
  val random = new Random(seed)
  val steady = 100
  val populationSize = 100
  RandomRegistry.random(random)

  def spawnSimulation(program : AggregateProgram, length : Int = 7, network : Option[GraphNeuralNetwork] = None) : (NetworkSimulator, Map[ID, Double]) = {
    val simulator = simulatorFactory.gridLike(
      GridSettings(3, 3, 50, 50),
      60,
      seeds = Seeds(seed, seed, seed)
    )
    network.foreach(simulator.addSensor("network", _))
    val networkSimulator = simulator.asInstanceOf[NetworkSimulator] //unsafe
    simulator.addSensor("source", 0.0f)
    simulator.chgSensorValue("source", Set(1), 1.0f)
    simulator.addSensor("initialState", Array(-1.0f, -1.0f, -1.0f, -1.0f))
    val ids = (0 to length).flatMap(_ => scala.util.Random.shuffle((0 until networkSimulator.ids.size).toList))
    ids foreach ( simulator.exec(program, program.main(), _))
    val results = simulator.exports().map { case (id, data) => id -> data.get.root[Double]() }
    (networkSimulator, results)
  }

  val hopCountProgram = new AggregateProgram with FieldUtils {
    override def main(): Any = rep(Double.PositiveInfinity) {
      value => mux(mid() == 0) { 0.0 } { minHoodPlus(nbr(value) + 1) }
    }
  }

  val (_, references) = spawnSimulation(hopCountProgram)


  def fitness(genotype : Genotype[DoubleGene], codec : GNNCodec) : Double = {
    val gnn = codec.loadFromGenotype(genotype)
    val (_, gnnResults) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
    val fitness = gnnResults.map { case (id, v) => references(id) - v }.map { value => Math.abs(value) }.sum
    fitness
  }

  val factory = codec.genotypeFactory()
  val runner : Engine[DoubleGene, lang.Double] = JeneticsFacade.doubleEngine[DoubleGene](genotype => fitness(genotype, codec), factory)
    .populationSize(populationSize)
    .survivorsSelector(new TournamentSelector(5))
    .offspringSelector(new RouletteWheelSelector())
    .alterers(
      new Mutator(0.315),
      new SinglePointCrossover(0.25))
    .minimizing()
    .build()
  val statistics = EvolutionStatistics.ofNumber[lang.Double]()
  val result =
    RandomRegistry.`with`(random, _ => {
      runner.stream()
        .limit(Limits.bySteadyFitness[lang.Double](steady))
        .peek(e => statistics.accept(e))
        .peek(e => println(s"Generation ${e.generation()}, mean fitness : ${e.population().map(_.fitness).asScala.map[Double](a => a).sum / e.population().size()}, best : ${e.bestFitness()}, worst : ${e.worstFitness()}"))
        .collect(EvolutionResult.toBestPhenotype[DoubleGene, lang.Double])
    })

  val gnn = codec.loadFromGenotype(result.genotype())
  val (_, gnnResult) = spawnSimulation(new ScafiHopCountGNN(), network = Some(gnn))
  result.genotype()
  println(gnnResult)
  println(references)

  val file = new FileOutputStream("result.xml")
  Writers.Genotype.write[lang.Double, DoubleGene, DoubleChromosome](file, result.genotype(), Writers.DoubleChromosome.writer())
}
