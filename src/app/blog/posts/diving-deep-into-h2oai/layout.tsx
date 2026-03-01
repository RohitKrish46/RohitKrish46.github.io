import { Metadata } from 'next'

export const metadata: Metadata = {
    title: 'Diving Deep Into H2O.ai | Rohit.ml',
    description: 'A hands-on look at H2O.ai\'s deep learning features, AutoML, and an MNIST digit classification walkthrough.',
    openGraph: {
        title: 'Diving Deep Into H2O.ai | Rohit.ml',
        description: 'A hands-on look at H2O.ai\'s deep learning features, AutoML, and an MNIST digit classification walkthrough.',
        type: 'article',
        publishedTime: '2020-04-18T00:00:00.000Z',
    },
    twitter: {
        card: 'summary_large_image',
        title: 'Diving Deep Into H2O.ai',
        description: 'A hands-on look at H2O.ai\'s deep learning features, AutoML, and an MNIST digit classification walkthrough.',
    }
}

export default function Layout({
    children,
}: {
    children: React.ReactNode
}) {
    return children
}
