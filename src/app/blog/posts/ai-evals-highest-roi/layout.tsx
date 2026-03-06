import { Metadata } from 'next'

export const metadata: Metadata = {
    title: 'AI Evals: The Highest ROI Activity for Building Great AI Products',
    description: 'A framework for building effective AI evaluations — why evals matter, the difference between model and product evals, and a 6-step process to get started.',
    openGraph: {
        title: 'AI Evals: The Highest ROI Activity for Building Great AI Products',
        description: 'A framework for building effective AI evaluations — why evals matter, the difference between model and product evals, and a 6-step process to get started.',
        type: 'article',
        publishedTime: '2026-03-05T00:00:00.000Z',
    },
    twitter: {
        card: 'summary_large_image',
        title: 'AI Evals: The Highest ROI Activity for Building Great AI Products',
        description: 'A framework for building effective AI evaluations — why evals matter, the difference between model and product evals, and a 6-step process to get started.',
    }
}

export default function Layout({
    children,
}: {
    children: React.ReactNode
}) {
    return children
}
