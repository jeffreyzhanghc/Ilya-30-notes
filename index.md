---
layout: default
---

<!-- Custom CSS -->
<style>
/* Modern type scale and colors */
:root {
  --primary: #2563eb;
  --primary-dark: #1e40af;
  --text-main: #1f2937;
  --text-secondary: #4b5563;
  --bg-paper: #ffffff;
  --bg-subtle: #f8fafc;
  --border: #e5e7eb;
}

/* Base styles */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--text-main);
  line-height: 1.6;
}

/* Container */
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

/* Header section */
.site-header {
  text-align: center;
  margin-bottom: 4rem;
  padding: 3rem 0;
  background: var(--bg-subtle);
  border-bottom: 1px solid var(--border);
}

.site-title {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--text-main);
  margin-bottom: 1rem;
  letter-spacing: -0.025em;
}

.site-description {
  font-size: 1.25rem;
  color: var(--text-secondary);
  max-width: 600px;
  margin: 0 auto;
}

/* Paper cards */
.paper-list {
  display: grid;
  gap: 2rem;
}

.paper-card {
  background: var(--bg-paper);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 2rem;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}

.paper-card:hover {
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.paper-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  width: 4px;
  background: var(--primary);
}

.paper-meta {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.paper-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: var(--text-main);
}

.paper-description {
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
}

/* Tags */
.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.tag {
  background: var(--bg-subtle);
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.875rem;
  color: var(--primary);
  border: 1px solid var(--primary);
}

/* Links */
.paper-links {
  display: flex;
  gap: 1rem;
}

.link-primary, .link-secondary {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.link-primary {
  background: var(--primary);
  color: white;
}

.link-primary:hover {
  background: var(--primary-dark);
}

.link-secondary {
  background: var(--bg-subtle);
  color: var(--text-secondary);
}

.link-secondary:hover {
  background: var(--border);
}
</style>

<header class="site-header">
  <h1 class="site-title">AI Paper Reviews</h1>
  <p class="site-description">Deep dives into Ilya Sutskever's recommended papers, exploring the foundations of modern AI development</p>
</header>

<div class="container">
  <div class="paper-list">
    <article class="paper-card">
      <div class="paper-meta">
        <span>December 16, 2024</span>
        <span>•</span>
        <span>Kaplan et al.</span>
        <span>•</span>
        <span>OpenAI</span>
      </div>
      <h2 class="paper-title">Scaling Laws for Neural Language Models</h2>
      <p class="paper-description">
        A comprehensive empirical analysis establishing fundamental power-law relationships between model performance and computational resources. This work provides crucial insights into the scalability of language models and guides modern AI development strategies.
      </p>
      <div class="tag-list">
        <span class="tag">Language Models</span>
        <span class="tag">Deep Learning</span>
        <span class="tag">Empirical Analysis</span>
      </div>
      <div class="paper-links">
        <a href="posts/scaling-laws.html" class="link-primary">
          Read Review
        </a>
        <a href="https://arxiv.org/abs/2001.08361" class="link-secondary" target="_blank" rel="noopener">
          Original Paper
        </a>
      </div>
    </article>

    <!-- Template for future papers -->
    <!-- Copy the article.paper-card structure and update content -->
  </div>
</div>