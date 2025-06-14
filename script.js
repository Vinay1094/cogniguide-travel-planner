document.addEventListener('DOMContentLoaded', () => {
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    const answerOutput = document.getElementById('answerOutput');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // Function to send question to the backend
    const askCogniGuide = async () => {
        const query = questionInput.value.trim(); // Get and trim the input
        if (!query) {
            alert('Please enter a question!');
            return;
        }

        // Clear previous answer and show loading indicator
        answerOutput.innerHTML = '<p>Thinking...</p>';
        loadingIndicator.style.display = 'block';
        askButton.disabled = true; // Disable button while processing

            try {
            const backendUrl = 'https://cogniguide.onrender.com'; // 

            const response = await fetch(backendUrl + '/ask', { // <-- Corrected fetch call
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            answerOutput.innerHTML = `<p>${data.answer}</p>`; // Display the answer
        } catch (error) {
            console.error('Error:', error);
            answerOutput.innerHTML = `<p style="color: red;">Error: Could not get a response. ${error.message || 'Please check the backend server.'}</p>`;
        } finally {
            loadingIndicator.style.display = 'none'; // Hide loading
            askButton.disabled = false; // Re-enable button
        }     


        
    };

    // Event Listeners
    askButton.addEventListener('click', askCogniGuide);

    // Allow pressing Enter in the textarea to submit the question
    questionInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) { // Check for Enter key without Shift
            event.preventDefault(); // Prevent new line in textarea
            askCogniGuide();
        }
    });
});