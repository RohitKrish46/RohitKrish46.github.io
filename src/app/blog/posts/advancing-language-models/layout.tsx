import { Metadata } from 'next'

export const metadata: Metadata = {
    title: 'Advancing Language Models | Rohit.ml',
    description: 'How Transformers and Self-Attention are advancing the frontiers of NLP — from statistical models to GPT-3, LLaMA, and BERT.',
    openGraph: {
        title: 'Advancing Language Models | Rohit.ml',
        description: 'How Transformers and Self-Attention are advancing the frontiers of NLP — from statistical models to GPT-3, LLaMA, and BERT.',
        type: 'article',
        publishedTime: '2023-03-28T00:00:00.000Z',
    },
    twitter: {
        card: 'summary_large_image',
        title: 'Advancing Language Models',
        description: 'How Transformers and Self-Attention are advancing the frontiers of NLP — from statistical models to GPT-3, LLaMA, and BERT.',
    }
}

export default function Layout({
    children,
}: {
    children: React.ReactNode
}) {
    return children
}
