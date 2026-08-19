import path from 'path';
import prisma from './prisma';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');

console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);

export const getTrainingFolder = async () => {
  const key = 'TRAINING_FOLDER';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  return trainingRoot as string;
};

export const getHFToken = async () => {
  const key = 'HF_TOKEN';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  return token;
};

export const getSamplePublicKey = async () => {
  const key = 'SAMPLE_PUBLIC_KEY';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let publicKey = '';
  if (row?.value && row.value !== '') {
    publicKey = row.value;
  }
  return publicKey;
};

export const getDatasetPassword = async () => {
  const key = 'DATASET_PASSWORD';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let password = '';
  if (row?.value && row.value !== '') {
    password = row.value;
  }
  return password;
};
