"""Unit tests for fitness evaluation."""


from genetic_mcp.fitness import FitnessEvaluator
from genetic_mcp.models import FitnessWeights, Idea


class TestFitnessEvaluator:
    """Test the FitnessEvaluator class."""

    def test_evaluator_creation(self):
        """Test creating a fitness evaluator."""
        evaluator = FitnessEvaluator()

        assert evaluator.weights.relevance == 0.4
        assert evaluator.weights.novelty == 0.3
        assert evaluator.weights.feasibility == 0.3

    def test_evaluator_with_custom_weights(self):
        """Test evaluator with custom weights."""
        weights = FitnessWeights(relevance=0.5, novelty=0.2, feasibility=0.3)
        evaluator = FitnessEvaluator(weights)

        assert evaluator.weights.relevance == 0.5
        assert evaluator.weights.novelty == 0.2
        assert evaluator.weights.feasibility == 0.3

    def test_add_embedding(self):
        """Test adding embeddings to cache."""
        evaluator = FitnessEvaluator()
        embedding = [0.1] * 1536  # Standard embedding size

        evaluator.add_embedding("idea-1", embedding)

        assert "idea-1" in evaluator.embeddings_cache
        assert evaluator.embeddings_cache["idea-1"] == embedding

    def test_calculate_relevance(self):
        """Test relevance calculation."""
        evaluator = FitnessEvaluator()

        # Create embeddings
        target_embedding = [1.0, 0.0, 0.0, 0.0]
        idea_embedding1 = [1.0, 0.0, 0.0, 0.0]  # Perfect match
        idea_embedding2 = [0.0, 1.0, 0.0, 0.0]  # Orthogonal

        idea1 = Idea(id="idea-1", content="Test 1")
        idea2 = Idea(id="idea-2", content="Test 2")

        evaluator.add_embedding("idea-1", idea_embedding1)
        evaluator.add_embedding("idea-2", idea_embedding2)

        relevance1 = evaluator._calculate_relevance(idea1, target_embedding)
        relevance2 = evaluator._calculate_relevance(idea2, target_embedding)

        # Perfect match should have relevance 1.0
        assert abs(relevance1 - 1.0) < 0.01
        # Orthogonal should have relevance 0.5 (normalized from 0)
        assert abs(relevance2 - 0.5) < 0.01

    def test_calculate_novelty(self):
        """Test novelty calculation."""
        evaluator = FitnessEvaluator()

        # Create ideas and embeddings
        ideas = [
            Idea(id="idea-1", content="Test 1"),
            Idea(id="idea-2", content="Test 2"),
            Idea(id="idea-3", content="Test 3"),
        ]

        # Similar embeddings
        evaluator.add_embedding("idea-1", [1.0, 0.0, 0.0, 0.0])
        evaluator.add_embedding("idea-2", [0.9, 0.1, 0.0, 0.0])
        evaluator.add_embedding("idea-3", [0.0, 0.0, 1.0, 0.0])  # Different

        novelty1 = evaluator._calculate_novelty(ideas[0], ideas)
        novelty3 = evaluator._calculate_novelty(ideas[2], ideas)

        # idea-3 should be more novel than idea-1
        assert novelty3 > novelty1

    def test_calculate_feasibility(self):
        """Test feasibility calculation."""
        evaluator = FitnessEvaluator()

        # Test various content types
        idea_short = Idea(id="1", content="Too short")
        idea_good = Idea(
            id="2",
            content="This is a well-structured idea with implementation details. "
                   "1. First step\n2. Second step\n3. Third step"
        )
        idea_long = Idea(id="3", content="word " * 600)  # Too long

        feasibility_short = evaluator._calculate_feasibility(idea_short)
        feasibility_good = evaluator._calculate_feasibility(idea_good)
        feasibility_long = evaluator._calculate_feasibility(idea_long)

        # Good idea should have higher feasibility
        assert feasibility_good > feasibility_short
        assert feasibility_good > feasibility_long
        assert 0 <= feasibility_short <= 1
        assert 0 <= feasibility_good <= 1
        assert 0 <= feasibility_long <= 1

    def test_calculate_fitness(self):
        """Test overall fitness calculation."""
        weights = FitnessWeights(relevance=0.5, novelty=0.3, feasibility=0.2)
        evaluator = FitnessEvaluator(weights)

        # Setup
        target_embedding = [1.0, 0.0, 0.0, 0.0]
        idea = Idea(id="test-idea", content="A good implementation approach with clear steps")
        evaluator.add_embedding("test-idea", [0.8, 0.2, 0.0, 0.0])

        all_ideas = [idea]

        fitness = evaluator.calculate_fitness(idea, all_ideas, target_embedding)

        assert 0 <= fitness <= 1
        assert idea.fitness == fitness
        assert "relevance" in idea.scores
        assert "novelty" in idea.scores
        assert "feasibility" in idea.scores

    def test_evaluate_population(self):
        """Test evaluating entire population."""
        evaluator = FitnessEvaluator()
        target_embedding = [1.0, 0.0, 0.0, 0.0]

        ideas = [
            Idea(id="1", content="First idea"),
            Idea(id="2", content="Second idea with more details"),
            Idea(id="3", content="Third idea"),
        ]

        # Add embeddings
        evaluator.add_embedding("1", [1.0, 0.0, 0.0, 0.0])
        evaluator.add_embedding("2", [0.7, 0.3, 0.0, 0.0])
        evaluator.add_embedding("3", [0.0, 1.0, 0.0, 0.0])

        evaluator.evaluate_population(ideas, target_embedding)

        # All ideas should have fitness scores
        for idea in ideas:
            assert idea.fitness > 0
            assert len(idea.scores) == 3

    def test_get_selection_probabilities(self):
        """Test getting selection probabilities."""
        evaluator = FitnessEvaluator()

        ideas = [
            Idea(id="1", content="A", fitness=0.1),
            Idea(id="2", content="B", fitness=0.3),
            Idea(id="3", content="C", fitness=0.6),
        ]

        probs = evaluator.get_selection_probabilities(ideas)

        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 0.001
        assert probs[2] > probs[1] > probs[0]  # Higher fitness = higher probability

    def test_tournament_select(self):
        """Test tournament selection."""
        evaluator = FitnessEvaluator()

        ideas = [
            Idea(id="1", content="A", fitness=0.1),
            Idea(id="2", content="B", fitness=0.3),
            Idea(id="3", content="C", fitness=0.9),
            Idea(id="4", content="D", fitness=0.5),
        ]

        # Run multiple tournaments
        winners = []
        for _ in range(10):
            winner = evaluator.tournament_select(ideas, tournament_size=2)
            winners.append(winner.id)

        # High fitness idea should win more often
        assert winners.count("3") > winners.count("1")
