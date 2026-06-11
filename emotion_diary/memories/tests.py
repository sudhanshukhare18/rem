import random
import time

from django.contrib.auth import get_user_model
from django.test import TestCase

from memories.models import Memory
from memories.services.memory_store import process_and_store_text
from memories.services.memory_search import search_memories


User = get_user_model()


class BulkMemoryStressTest(TestCase):
    """
    Stress test memory system with many memories.
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username="stressuser",
            password="testpass123"
        )

    def generate_memory_samples(self):
        """
        Generate diverse emotional memories.
        """

        templates = [
            "Today was amazing and I felt so happy.",
            "I am feeling sad and lonely tonight.",
            "This morning was peaceful and calm.",
            "I had an exciting adventure with friends.",
            "I feel anxious about my future.",
            "Life feels beautiful and meaningful.",
            "I got angry during the meeting today.",
            "I miss my childhood memories.",
            "I feel motivated to achieve my goals.",
            "Today was emotionally exhausting.",
        ]

        return [
            random.choice(templates)
            for _ in range(100)
        ]

    def test_bulk_memory_storage_and_search(self):
        """
        Store 100 memories and test semantic search.
        """

        memories = self.generate_memory_samples()

        start_time = time.time()

        # Store memories
        for text in memories:

            memory = process_and_store_text(
                text=text,
                user=self.user
            )

            self.assertIsNotNone(memory)

        end_time = time.time()

        total_time = end_time - start_time

        print(f"\n✅ Stored 100 memories in {total_time:.2f} seconds")

        # Verify database count
        self.assertEqual(
            Memory.objects.count(),
            100
        )

        # Run semantic search
        result = search_memories(
            query="happy peaceful life",
            user=self.user,
            top_k=5
        )

        print("\n🔍 SEARCH SUMMARY:")
        print(result["summary"])

        print("\n📌 TOP MATCHES:")

        for memory in result["matches"]:

            print(
                f"- {memory.text[:80]} "
                f"({memory.emotion})"
            )

        # Assertions
        self.assertIn("summary", result)

        self.assertIn("matches", result)

        self.assertGreater(
            len(result["matches"]),
            0
        )